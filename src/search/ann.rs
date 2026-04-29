// search/ann.rs — approximate nearest-neighbour search orchestration.
use std::{cmp::Ordering, collections::HashMap, sync::Arc, time::Instant};

use futures::future::join_all;
use tracing::instrument;

use crate::{
    collections::CollectionManager,
    embeddings::EmbeddingClient,
    error::{VectorError, VectorResult},
    search::{
        filters::apply_filter,
        rerank::{apply_reranker_config, reranker_needs_vectors},
    },
    types::{DistanceMetric, SearchMetrics, SearchQuery, SearchResponse, SearchResult},
};

/// Core ANN search service.
pub struct AnnSearcher {
    /// Collection and persistence coordinator used during search.
    pub collection_manager: Arc<CollectionManager>,
}

impl AnnSearcher {
    /// Create a new ANN searcher.
    pub fn new(collection_manager: Arc<CollectionManager>) -> Self {
        Self { collection_manager }
    }

    /// Execute a nearest-neighbour search and return filtered, ranked results with metrics.
    #[instrument(skip(self, query))]
    pub async fn search(&self, query: SearchQuery) -> VectorResult<SearchResponse> {
        query.validate()?;

        let started = Instant::now();
        let collection = self
            .collection_manager
            .get_collection(&query.collection)
            .await?;
        if query.vector.len() != collection.dimensions {
            return Err(VectorError::DimensionMismatch {
                expected: collection.dimensions,
                got: query.vector.len(),
            });
        }

        let candidate_limit = query.top_k.saturating_mul(2).max(query.top_k);
        let ef_search = query
            .ef_search
            .unwrap_or(self.collection_manager.config.ef_search);

        let raw_candidates = {
            let indexes = self.collection_manager.indexes.read().await;
            let index = indexes
                .get(&query.collection)
                .ok_or_else(|| VectorError::NotFound {
                    entity: "collection".into(),
                    id: query.collection.clone(),
                })?;
            index.search(&query.vector, candidate_limit, ef_search)?
        };

        let candidate_ids = raw_candidates
            .iter()
            .map(|(internal_id, _)| *internal_id)
            .collect::<Vec<_>>();
        let records = self
            .collection_manager
            .store
            .bulk_internal_to_uuid(&query.collection, &candidate_ids)
            .await?;
        let mut records_by_id: HashMap<usize, crate::types::VectorRecord> =
            records.into_iter().collect();

        let needs_vectors =
            query.include_vectors || reranker_needs_vectors(query.reranker.as_ref());
        let mut results = Vec::new();
        for (internal_id, distance) in raw_candidates {
            let record = match records_by_id.remove(&internal_id) {
                Some(record) => record,
                None => continue,
            };

            if let Some(filter) = &query.filter {
                if !apply_filter(filter, &record.metadata) {
                    continue;
                }
            }

            let vector = if needs_vectors {
                Some(
                    self.collection_manager
                        .read_vector_by_internal_id(&query.collection, internal_id)
                        .await?,
                )
            } else {
                None
            };

            results.push(SearchResult {
                id: record.id,
                score: normalize_distance(distance, collection.distance),
                vector,
                metadata: if query.include_metadata {
                    record.metadata.clone()
                } else {
                    serde_json::Value::Null
                },
                text: record.text.clone(),
                created_at: record.created_at,
            });
        }

        let post_filter_count = results.len();
        let mut results =
            apply_reranker_config(&query.vector, results, query.reranker.as_ref()).await?;
        results.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
        });
        results.truncate(query.top_k);

        if !query.include_vectors {
            for result in &mut results {
                result.vector = None;
            }
        }

        Ok(SearchResponse {
            metrics: SearchMetrics {
                query_vector_dims: query.vector.len(),
                candidates_evaluated: candidate_ids.len(),
                post_filter_count,
                latency_us: started.elapsed().as_micros() as u64,
            },
            results,
        })
    }

    /// Embed free-form text and execute ANN search.
    #[instrument(skip(self, embedding_client, text))]
    pub async fn search_by_text(
        &self,
        collection: &str,
        text: &str,
        top_k: usize,
        embedding_client: &EmbeddingClient,
    ) -> VectorResult<SearchResponse> {
        let vector = embedding_client.embed_one(text).await?;
        self.search(SearchQuery {
            collection: collection.to_string(),
            vector,
            top_k,
            filter: None,
            include_vectors: false,
            include_metadata: true,
            ef_search: None,
            reranker: None,
        })
        .await
    }

    /// Execute multiple ANN queries concurrently.
    #[instrument(skip(self, queries))]
    pub async fn batch_search(
        &self,
        queries: Vec<SearchQuery>,
    ) -> VectorResult<Vec<SearchResponse>> {
        let handles = queries
            .into_iter()
            .map(|query| {
                let searcher = AnnSearcher {
                    collection_manager: Arc::clone(&self.collection_manager),
                };
                tokio::task::spawn(async move { searcher.search(query).await })
            })
            .collect::<Vec<_>>();

        let mut responses = Vec::with_capacity(handles.len());
        for handle in join_all(handles).await {
            let response = handle.map_err(|err| {
                VectorError::SearchError(format!("ANN batch task failed: {err}"))
            })??;
            responses.push(response);
        }

        Ok(responses)
    }
}

fn normalize_distance(distance: f32, metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Cosine | DistanceMetric::Euclidean => {
            (1.0 / (1.0 + distance.max(0.0))).clamp(0.0, 1.0)
        }
        DistanceMetric::DotProduct => (1.0 / (1.0 + distance.exp())).clamp(0.0, 1.0),
    }
}
