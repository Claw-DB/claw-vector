// embeddings/client.rs — gRPC client to the Python embedding microservice.
use std::time::Duration;
use tonic::transport::Channel;
use tracing::instrument;

use crate::{
    config::VectorConfig,
    embeddings::{
        cache::EmbeddingCache,
        types::{EmbeddingRequest, EmbeddingResponse, EmbedVector},
    },
    error::{VectorError, VectorResult},
    grpc::proto::{
        embedding_service_client::EmbeddingServiceClient, EmbedRequest, HealthRequest,
        ModelInfoRequest,
    },
};

/// gRPC client that calls the Python embedding service and caches results locally.
pub struct EmbeddingClient {
    inner: EmbeddingServiceClient<Channel>,
    cache: EmbeddingCache,
    timeout: Duration,
    batch_size: usize,
}

impl EmbeddingClient {
    /// Connect to the embedding service and return a client with a local LRU cache.
    pub async fn connect(config: &VectorConfig) -> VectorResult<Self> {
        let channel = Channel::from_shared(config.embedding_service_url.clone())
            .map_err(|e| VectorError::Embedding(e.to_string()))?
            .connect()
            .await
            .map_err(|e| VectorError::Embedding(e.to_string()))?;
        Ok(EmbeddingClient {
            inner: EmbeddingServiceClient::new(channel),
            cache: EmbeddingCache::new(config.cache_size),
            timeout: Duration::from_millis(config.embedding_timeout_ms),
            batch_size: config.batch_size,
        })
    }

    /// Embed a list of texts, serving cached results where available.
    #[instrument(skip(self, request))]
    pub async fn embed(&mut self, request: EmbeddingRequest) -> VectorResult<EmbeddingResponse> {
        let _ = self.timeout; // applied per-call in future via request interceptor

        let mut cached: std::collections::HashMap<usize, Vec<f32>> = std::collections::HashMap::new();
        let mut uncached_idx: Vec<usize> = Vec::new();
        let mut uncached_texts: Vec<String> = Vec::new();

        for (i, text) in request.texts.iter().enumerate() {
            if let Some(v) = self.cache.get(text) {
                cached.insert(i, v);
            } else {
                uncached_idx.push(i);
                uncached_texts.push(text.clone());
            }
        }

        let mut all: Vec<Option<Vec<f32>>> = vec![None; request.texts.len()];
        for (i, v) in cached { all[i] = Some(v); }

        let start = std::time::Instant::now();
        for (chunk_num, chunk_idx) in uncached_idx.chunks(self.batch_size).enumerate() {
            let offset = chunk_num * self.batch_size;
            let batch_texts: Vec<String> = chunk_idx
                .iter()
                .enumerate()
                .map(|(j, _)| uncached_texts[offset + j].clone())
                .collect();

            let resp = self
                .inner
                .embed(tonic::Request::new(EmbedRequest {
                    texts: batch_texts.clone(),
                    model_name: request.model_name.clone(),
                    normalize: request.normalize,
                }))
                .await
                .map_err(VectorError::Grpc)?
                .into_inner();

            for (j, vec_proto) in resp.vectors.into_iter().enumerate() {
                self.cache.insert(batch_texts[j].clone(), vec_proto.values.clone());
                all[chunk_idx[j]] = Some(vec_proto.values);
            }
        }

        let latency_ms = start.elapsed().as_millis() as i64;
        let vectors: Vec<EmbedVector> = all.into_iter().map(|opt| {
            let values = opt.unwrap_or_default();
            let dimensions = values.len() as i32;
            EmbedVector { values, dimensions }
        }).collect();

        Ok(EmbeddingResponse { vectors, model_name: request.model_name, latency_ms })
    }

    /// Health-check the embedding service.
    #[instrument(skip(self))]
    pub async fn health(&mut self) -> VectorResult<bool> {
        let resp = self.inner.health(tonic::Request::new(HealthRequest {}))
            .await.map_err(VectorError::Grpc)?.into_inner();
        Ok(resp.ready)
    }

    /// Return metadata about the currently loaded model.
    #[instrument(skip(self))]
    pub async fn model_info(&mut self) -> VectorResult<serde_json::Value> {
        let resp = self.inner.model_info(tonic::Request::new(ModelInfoRequest {}))
            .await.map_err(VectorError::Grpc)?.into_inner();
        Ok(serde_json::json!({
            "model_name": resp.model_name,
            "dimensions": resp.dimensions,
            "max_sequence_length": resp.max_sequence_length,
            "device": resp.device,
        }))
    }
}
