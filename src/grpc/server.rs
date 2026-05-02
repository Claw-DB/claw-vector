use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Instant,
};

use sha2::{Digest, Sha256};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{
    metadata::MetadataValue, service::interceptor::InterceptedService, transport::Server, Request,
    Response, Status,
};

use crate::{
    config::VectorConfig,
    engine::VectorEngine,
    grpc::proto::{
        embedding_service_server::{EmbeddingService, EmbeddingServiceServer},
        vector_service_server::{VectorService, VectorServiceServer},
        CollectionInfo, CollectionStatsResponse, CreateCollectionRequest, DeleteCollectionRequest,
        DeleteResult, EmbedRequest, EmbedResponse, HealthRequest, HealthResponse,
        ListCollectionsResponse, ListRequest, ModelInfoRequest, ModelInfoResponse,
        SearchMetricsProto, SearchRequest, SearchResponseProto, StatsRequest, UpsertResult,
        UpsertVectorRequest,
    },
    types::{DistanceMetric, SearchQuery},
    VectorError,
};

const WORKSPACE_HEADER: &str = "x-claw-workspace-id";
const API_KEY_HEADER: &str = "x-claw-api-key";
const TRACE_HEADER: &str = "x-trace-id";

#[derive(Clone)]
struct WorkspaceId(String);

#[derive(Clone)]
struct TraceId(String);

/// Minimal pass-through embedding service stub for local Rust gRPC server mode.
pub struct EmbeddingServiceImpl;

#[tonic::async_trait]
impl EmbeddingService for EmbeddingServiceImpl {
    async fn embed(
        &self,
        _request: Request<EmbedRequest>,
    ) -> Result<Response<EmbedResponse>, Status> {
        Err(Status::unimplemented(
            "Embed is handled by the Python embedding service",
        ))
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse {
            ready: false,
            model_name: String::new(),
            model_load_time_ms: 0,
        }))
    }

    async fn model_info(
        &self,
        _request: Request<ModelInfoRequest>,
    ) -> Result<Response<ModelInfoResponse>, Status> {
        Err(Status::unimplemented(
            "ModelInfo is handled by the Python embedding service",
        ))
    }

    type EmbedStreamStream = ReceiverStream<Result<EmbedResponse, Status>>;

    async fn embed_stream(
        &self,
        _request: Request<tonic::Streaming<EmbedRequest>>,
    ) -> Result<Response<Self::EmbedStreamStream>, Status> {
        Err(Status::unimplemented(
            "EmbedStream is handled by the Python embedding service",
        ))
    }
}

#[derive(Clone)]
struct ServerState {
    default_workspace_id: String,
    require_auth: bool,
    default_rate_limit_rps: u32,
    api_keys: Arc<HashMap<String, String>>,
    workspace_rate_limits: Arc<HashMap<String, u32>>,
    buckets: Arc<Mutex<HashMap<String, TokenBucket>>>,
}

#[derive(Clone)]
struct AuthRateTraceInterceptor {
    state: Arc<ServerState>,
}

impl tonic::service::Interceptor for AuthRateTraceInterceptor {
    fn call(&mut self, mut request: Request<()>) -> Result<Request<()>, Status> {
        let workspace_id = request
            .metadata()
            .get(WORKSPACE_HEADER)
            .and_then(|value| value.to_str().ok())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| self.state.default_workspace_id.clone());

        if self.state.require_auth {
            let api_key = request
                .metadata()
                .get(API_KEY_HEADER)
                .and_then(|value| value.to_str().ok())
                .ok_or_else(|| Status::unauthenticated("missing x-claw-api-key"))?;

            let hashed = hash_api_key(api_key);
            let valid_workspace = self
                .state
                .api_keys
                .get(&hashed)
                .map(|ws| ws == &workspace_id)
                .unwrap_or(false);
            if !valid_workspace {
                return Err(Status::unauthenticated("invalid API key"));
            }
        }

        let rate_limit = self
            .state
            .workspace_rate_limits
            .get(&workspace_id)
            .copied()
            .unwrap_or(self.state.default_rate_limit_rps)
            .max(1);
        {
            let mut buckets = self
                .state
                .buckets
                .lock()
                .map_err(|_| Status::internal("rate limiter lock poisoned"))?;
            let bucket = buckets
                .entry(workspace_id.clone())
                .or_insert_with(|| TokenBucket::new(rate_limit));
            bucket.rate_limit_rps = rate_limit;
            if !bucket.try_consume(1.0) {
                return Err(Status::resource_exhausted("rate limit exceeded"));
            }
        }

        request.extensions_mut().insert(WorkspaceId(workspace_id));
        request
            .extensions_mut()
            .insert(TraceId(format!("trace-{}", uuid::Uuid::new_v4())));
        Ok(request)
    }
}

#[derive(Debug)]
struct TokenBucket {
    tokens: f64,
    last_refill: Instant,
    rate_limit_rps: u32,
}

impl TokenBucket {
    fn new(rate_limit_rps: u32) -> Self {
        let rate = rate_limit_rps.max(1) as f64;
        Self {
            tokens: rate,
            last_refill: Instant::now(),
            rate_limit_rps,
        }
    }

    fn try_consume(&mut self, cost: f64) -> bool {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        let rate = self.rate_limit_rps.max(1) as f64;
        self.tokens = (self.tokens + elapsed * rate).min(rate);
        self.last_refill = now;
        if self.tokens >= cost {
            self.tokens -= cost;
            true
        } else {
            false
        }
    }
}

/// VectorService gRPC implementation backed by [`VectorEngine`].
pub struct VectorServiceImpl {
    engine: Arc<VectorEngine>,
}

impl VectorServiceImpl {
    fn workspace_from_request<T>(&self, request: &Request<T>) -> String {
        request
            .extensions()
            .get::<WorkspaceId>()
            .map(|value| value.0.clone())
            .unwrap_or_else(|| self.engine.config.default_workspace_id.clone())
    }

    fn trace_from_request<T>(
        &self,
        request: &Request<T>,
    ) -> Option<MetadataValue<tonic::metadata::Ascii>> {
        request
            .extensions()
            .get::<TraceId>()
            .and_then(|value| MetadataValue::try_from(value.0.as_str()).ok())
    }
}

#[tonic::async_trait]
impl VectorService for VectorServiceImpl {
    async fn create_collection(
        &self,
        request: Request<CreateCollectionRequest>,
    ) -> Result<Response<CollectionInfo>, Status> {
        let trace = self.trace_from_request(&request);
        let workspace_id = self.workspace_from_request(&request);
        let req = request.into_inner();
        let metric = parse_distance_metric(&req.distance_metric)?;
        let created = self
            .engine
            .create_collection_in_workspace(
                &workspace_id,
                &req.name,
                req.dimensions as usize,
                metric,
            )
            .await
            .map_err(Status::from)?;
        let info = collection_to_proto(&created);
        let mut response = Response::new(info);
        if let Some(trace_id) = trace {
            response.metadata_mut().insert(TRACE_HEADER, trace_id);
        }
        Ok(response)
    }

    async fn delete_collection(
        &self,
        request: Request<DeleteCollectionRequest>,
    ) -> Result<Response<DeleteResult>, Status> {
        let trace = self.trace_from_request(&request);
        let workspace_id = self.workspace_from_request(&request);
        let req = request.into_inner();
        let stats = self
            .engine
            .collections
            .store
            .collection_stats(&workspace_id, &req.name)
            .await
            .ok();
        self.engine
            .delete_collection_in_workspace(&workspace_id, &req.name)
            .await
            .map_err(Status::from)?;
        let mut response = Response::new(DeleteResult {
            records_removed: stats.map(|value| value.vector_count).unwrap_or(0),
        });
        if let Some(trace_id) = trace {
            response.metadata_mut().insert(TRACE_HEADER, trace_id);
        }
        Ok(response)
    }

    async fn upsert_vector(
        &self,
        request: Request<UpsertVectorRequest>,
    ) -> Result<Response<UpsertResult>, Status> {
        let trace = self.trace_from_request(&request);
        let workspace_id = self.workspace_from_request(&request);
        let req = request.into_inner();
        let metadata = if req.metadata_json.trim().is_empty() {
            serde_json::json!({})
        } else {
            serde_json::from_str(&req.metadata_json)
                .map_err(|err| Status::invalid_argument(format!("invalid metadata_json: {err}")))?
        };

        let id = if !req.vector.is_empty() {
            self.engine
                .upsert_vector_in_workspace(&workspace_id, &req.collection, req.vector, metadata)
                .await
                .map_err(Status::from)?
        } else if !req.text.trim().is_empty() {
            self.engine
                .upsert_in_workspace(&workspace_id, &req.collection, &req.text, metadata)
                .await
                .map_err(Status::from)?
        } else {
            return Err(Status::invalid_argument(
                "either vector or text must be provided",
            ));
        };

        let mut response = Response::new(UpsertResult { id: id.to_string() });
        if let Some(trace_id) = trace {
            response.metadata_mut().insert(TRACE_HEADER, trace_id);
        }
        Ok(response)
    }

    async fn search_vectors(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponseProto>, Status> {
        let trace = self.trace_from_request(&request);
        let workspace_id = self.workspace_from_request(&request);
        let req = request.into_inner();
        let filter =
            if req.filter_json.trim().is_empty() {
                None
            } else {
                Some(serde_json::from_str(&req.filter_json).map_err(|err| {
                    Status::invalid_argument(format!("invalid filter_json: {err}"))
                })?)
            };

        let query = if !req.vector.is_empty() {
            SearchQuery {
                collection: req.collection,
                vector: req.vector,
                top_k: req.top_k.max(1) as usize,
                filter,
                include_vectors: req.include_vectors,
                include_metadata: req.include_metadata,
                ef_search: None,
                reranker: None,
            }
        } else if !req.text.trim().is_empty() {
            let embedded = self
                .engine
                .embedding_client
                .embed_one(&req.text)
                .await
                .map_err(Status::from)?;
            SearchQuery {
                collection: req.collection,
                vector: embedded,
                top_k: req.top_k.max(1) as usize,
                filter,
                include_vectors: req.include_vectors,
                include_metadata: req.include_metadata,
                ef_search: None,
                reranker: None,
            }
        } else {
            return Err(Status::invalid_argument(
                "either vector or text must be provided",
            ));
        };

        let response = self
            .engine
            .search_in_workspace(&workspace_id, query)
            .await
            .map_err(Status::from)?;

        let proto = SearchResponseProto {
            results: response
                .results
                .into_iter()
                .map(|result| crate::grpc::proto::SearchHit {
                    id: result.id.to_string(),
                    score: result.score,
                    vector: result.vector.unwrap_or_default(),
                    metadata_json: if result.metadata.is_null() {
                        "{}".to_string()
                    } else {
                        serde_json::to_string(&result.metadata).unwrap_or_else(|_| "{}".to_string())
                    },
                    text: result.text.unwrap_or_default(),
                })
                .collect(),
            metrics: Some(SearchMetricsProto {
                query_vector_dims: response.metrics.query_vector_dims as u32,
                candidates_evaluated: response.metrics.candidates_evaluated as u32,
                post_filter_count: response.metrics.post_filter_count as u32,
                latency_us: response.metrics.latency_us,
            }),
        };

        let mut response = Response::new(proto);
        if let Some(trace_id) = trace {
            response.metadata_mut().insert(TRACE_HEADER, trace_id);
        }
        Ok(response)
    }

    async fn get_collection_stats(
        &self,
        request: Request<StatsRequest>,
    ) -> Result<Response<CollectionStatsResponse>, Status> {
        let trace = self.trace_from_request(&request);
        let workspace_id = self.workspace_from_request(&request);
        let req = request.into_inner();
        let collection = self
            .engine
            .collections
            .get_collection(&workspace_id, &req.collection)
            .await
            .map_err(Status::from)?;

        let response = CollectionStatsResponse {
            vector_count: collection.vector_count,
            index_type: format!("{:?}", collection.index_type).to_lowercase(),
            dimensions: collection.dimensions as u32,
            last_modified_at: collection.created_at.timestamp_millis(),
        };

        let mut response = Response::new(response);
        if let Some(trace_id) = trace {
            response.metadata_mut().insert(TRACE_HEADER, trace_id);
        }
        Ok(response)
    }

    async fn list_collections(
        &self,
        request: Request<ListRequest>,
    ) -> Result<Response<ListCollectionsResponse>, Status> {
        let trace = self.trace_from_request(&request);
        let workspace_id = self.workspace_from_request(&request);
        let req = request.into_inner();
        let page_size = req.page_size.clamp(1, 500) as usize;
        let page = req.page.max(1) as usize;

        let collections = self
            .engine
            .list_collections_in_workspace(&workspace_id)
            .await
            .map_err(Status::from)?;
        let total = collections.len();
        let start = page_size.saturating_mul(page - 1);
        let page_items = collections
            .into_iter()
            .skip(start)
            .take(page_size)
            .map(|collection| collection_to_proto(&collection))
            .collect::<Vec<_>>();

        let mut response = Response::new(ListCollectionsResponse {
            collections: page_items,
            page: page as u32,
            page_size: page_size as u32,
            total: total as u32,
        });
        if let Some(trace_id) = trace {
            response.metadata_mut().insert(TRACE_HEADER, trace_id);
        }
        Ok(response)
    }
}

fn collection_to_proto(collection: &crate::types::Collection) -> CollectionInfo {
    CollectionInfo {
        id: format!("{}:{}", collection.workspace_id, collection.name),
        name: collection.name.clone(),
        dimensions: collection.dimensions as u32,
        distance_metric: format!("{:?}", collection.distance).to_lowercase(),
        index_type: format!("{:?}", collection.index_type).to_lowercase(),
        vector_count: collection.vector_count,
        last_modified_at: collection.created_at.timestamp_millis(),
    }
}

#[allow(clippy::result_large_err)]
fn parse_distance_metric(raw: &str) -> Result<DistanceMetric, Status> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "cosine" => Ok(DistanceMetric::Cosine),
        "euclidean" => Ok(DistanceMetric::Euclidean),
        "dot" | "dot_product" => Ok(DistanceMetric::DotProduct),
        _ => Err(Status::invalid_argument("unsupported distance metric")),
    }
}

fn hash_api_key(api_key: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(api_key.as_bytes());
    hex::encode(hasher.finalize())
}

async fn load_server_state(config: &VectorConfig) -> Result<Arc<ServerState>, VectorError> {
    let store = crate::store::sqlite::VectorStore::new(&config.api_key_store_path).await?;
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS api_keys (key_hash TEXT PRIMARY KEY, workspace_id TEXT NOT NULL, created_at TEXT NOT NULL, revoked INTEGER NOT NULL DEFAULT 0)",
    )
    .execute(store.pool())
    .await?;

    let key_rows = sqlx::query_as::<_, (String, String)>(
        "SELECT key_hash, workspace_id FROM api_keys WHERE revoked = 0",
    )
    .fetch_all(store.pool())
    .await?;
    let api_keys = key_rows.into_iter().collect::<HashMap<_, _>>();

    let mut workspace_rate_limits = HashMap::new();
    let has_rate_table = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='workspace_rate_limits'",
    )
    .fetch_one(store.pool())
    .await
    .unwrap_or(0)
        > 0;
    if has_rate_table {
        let rows = sqlx::query_as::<_, (String, i64)>(
            "SELECT workspace_id, rate_limit_rps FROM workspace_rate_limits",
        )
        .fetch_all(store.pool())
        .await
        .unwrap_or_default();
        for (workspace_id, rps) in rows {
            workspace_rate_limits.insert(workspace_id, (rps as u32).max(1));
        }
    }

    Ok(Arc::new(ServerState {
        default_workspace_id: config.default_workspace_id.clone(),
        require_auth: config.require_auth,
        default_rate_limit_rps: config.rate_limit_rps.max(1),
        api_keys: Arc::new(api_keys),
        workspace_rate_limits: Arc::new(workspace_rate_limits),
        buckets: Arc::new(Mutex::new(HashMap::new())),
    }))
}

/// Start the Rust gRPC server with auth, rate limiting, and trace-id interception.
pub async fn serve(addr: std::net::SocketAddr) -> Result<(), Box<dyn std::error::Error>> {
    let config = VectorConfig::from_env();
    let state = load_server_state(&config).await?;
    let engine = Arc::new(VectorEngine::new(config.clone()).await?);

    let interceptor = AuthRateTraceInterceptor { state };
    let embedding = EmbeddingServiceServer::new(EmbeddingServiceImpl);
    let vector = VectorServiceServer::new(VectorServiceImpl { engine });

    Server::builder()
        .add_service(InterceptedService::new(embedding, interceptor.clone()))
        .add_service(InterceptedService::new(vector, interceptor))
        .serve(addr)
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tonic::service::Interceptor;

    fn interceptor_for_test(require_auth: bool, rate_limit: u32) -> AuthRateTraceInterceptor {
        let mut api_keys = HashMap::new();
        api_keys.insert(hash_api_key("valid-key"), "ws-test".to_string());
        AuthRateTraceInterceptor {
            state: Arc::new(ServerState {
                default_workspace_id: "default".to_string(),
                require_auth,
                default_rate_limit_rps: rate_limit,
                api_keys: Arc::new(api_keys),
                workspace_rate_limits: Arc::new(HashMap::new()),
                buckets: Arc::new(Mutex::new(HashMap::new())),
            }),
        }
    }

    #[test]
    fn interceptor_rejects_invalid_api_key() {
        let mut interceptor = interceptor_for_test(true, 100);
        let mut request = Request::new(());
        request.metadata_mut().insert(
            WORKSPACE_HEADER,
            MetadataValue::try_from("ws-test").unwrap(),
        );
        request.metadata_mut().insert(
            API_KEY_HEADER,
            MetadataValue::try_from("wrong-key").unwrap(),
        );

        let result = interceptor.call(request);
        assert!(matches!(result, Err(status) if status.code() == tonic::Code::Unauthenticated));
    }

    #[test]
    fn interceptor_applies_workspace_rate_limit() {
        let mut interceptor = interceptor_for_test(false, 100);
        let mut last: Result<Request<()>, Status> = Ok(Request::new(()));
        for _ in 0..101 {
            let mut request = Request::new(());
            request.metadata_mut().insert(
                WORKSPACE_HEADER,
                MetadataValue::try_from("ws-test").unwrap(),
            );
            last = interceptor.call(request);
        }

        assert!(matches!(last, Err(status) if status.code() == tonic::Code::ResourceExhausted));
    }
}
