// embeddings/client.rs — gRPC client to the Python embedding microservice.
use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use tokio::{sync::Mutex, time::sleep};
use tonic::{
    transport::{Channel, Endpoint},
    Request,
};
use tracing::instrument;

use crate::{
    config::VectorConfig,
    embeddings::{
        cache::{EmbeddingCache, EmbeddingCacheStats},
        types::ModelInfo,
    },
    error::{VectorError, VectorResult},
    grpc::proto::{
        embedding_service_client::EmbeddingServiceClient, EmbedRequest, HealthRequest,
        ModelInfoRequest,
    },
};

/// Abstraction over text embedding providers.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Embed a list of texts.
    async fn embed(&self, texts: Vec<String>) -> VectorResult<Vec<Vec<f32>>>;

    /// Embed a single text.
    async fn embed_one(&self, text: &str) -> VectorResult<Vec<f32>>;

    /// Check service readiness.
    async fn health_check(&self) -> VectorResult<bool>;

    /// Return metadata about the active model.
    async fn model_info(&self) -> VectorResult<ModelInfo>;

    /// Return cache statistics when available.
    async fn cache_stats(&self) -> Option<EmbeddingCacheStats> {
        None
    }
}

/// gRPC client that calls the Python embedding service and caches results locally.
pub struct EmbeddingClient {
    /// Underlying tonic client.
    pub client: EmbeddingServiceClient<Channel>,
    /// Runtime configuration.
    pub config: VectorConfig,
    /// Shared embedding cache.
    pub cache: Arc<Mutex<EmbeddingCache>>,
}

impl EmbeddingClient {
    /// Connect to the embedding service, initialize the cache, and verify readiness.
    pub async fn new(config: &VectorConfig) -> VectorResult<Self> {
        let timeout = Duration::from_millis(config.embedding_timeout_ms);
        let endpoint = Endpoint::from_shared(config.embedding_service_url.clone())
            .map_err(|err| VectorError::Embedding(format!("invalid embedding URL: {err}")))?
            .connect_timeout(timeout)
            .timeout(timeout);
        let channel = endpoint.connect().await.map_err(|err| {
            VectorError::Embedding(format!("failed to connect to embedding service: {err}"))
        })?;

        let client = EmbeddingClient {
            client: EmbeddingServiceClient::new(channel),
            config: config.clone(),
            cache: Arc::new(Mutex::new(EmbeddingCache::new(config.cache_size))),
        };

        let mut delay = Duration::from_millis(100);
        for attempt in 0..3 {
            match client.health_check().await {
                Ok(true) => return Ok(client),
                Ok(false) if attempt < 2 => sleep(delay).await,
                Err(_) if attempt < 2 => sleep(delay).await,
                Ok(false) => {
                    return Err(VectorError::Embedding(
                        "embedding service is reachable but not ready".into(),
                    ))
                }
                Err(err) => return Err(err),
            }
            delay *= 2;
        }

        Err(VectorError::Embedding(
            "embedding service readiness check failed".into(),
        ))
    }

    /// Alias for [`EmbeddingClient::new`].
    pub async fn connect(config: &VectorConfig) -> VectorResult<Self> {
        Self::new(config).await
    }

    /// Return the current cache statistics snapshot.
    pub async fn cache_stats_snapshot(&self) -> EmbeddingCacheStats {
        self.cache.lock().await.stats()
    }

    /// Embed a list of texts.
    pub async fn embed(&self, texts: Vec<String>) -> VectorResult<Vec<Vec<f32>>> {
        <Self as EmbeddingProvider>::embed(self, texts).await
    }

    /// Embed a single text.
    pub async fn embed_one(&self, text: &str) -> VectorResult<Vec<f32>> {
        <Self as EmbeddingProvider>::embed_one(self, text).await
    }

    /// Health-check the embedding service.
    pub async fn health_check(&self) -> VectorResult<bool> {
        <Self as EmbeddingProvider>::health_check(self).await
    }

    /// Return model metadata from the embedding service.
    pub async fn model_info(&self) -> VectorResult<ModelInfo> {
        <Self as EmbeddingProvider>::model_info(self).await
    }
}

#[async_trait]
impl EmbeddingProvider for EmbeddingClient {
    /// Embed a list of texts, serving cached results where available.
    #[instrument(skip(self, texts))]
    async fn embed(&self, texts: Vec<String>) -> VectorResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut outputs: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut uncached = Vec::<(usize, String)>::new();

        {
            let mut cache = self.cache.lock().await;
            for (index, text) in texts.iter().enumerate() {
                if let Some(vector) = cache.get(text) {
                    outputs[index] = Some(vector);
                } else {
                    uncached.push((index, text.clone()));
                }
            }
        }

        let mut fresh_embeddings = Vec::<(String, Vec<f32>)>::new();
        let batch_size = self.config.batch_size.max(1);
        let mut client = self.client.clone();
        for chunk in uncached.chunks(batch_size) {
            let batch_texts = chunk
                .iter()
                .map(|(_, text)| text.clone())
                .collect::<Vec<_>>();
            let mut request = Request::new(EmbedRequest {
                texts: batch_texts.clone(),
                model_name: String::new(),
                normalize: true,
            });
            request.set_timeout(Duration::from_millis(self.config.embedding_timeout_ms));

            let response = client
                .embed(request)
                .await
                .map_err(VectorError::from)?
                .into_inner();

            if response.vectors.len() != batch_texts.len() {
                return Err(VectorError::Embedding(format!(
                    "embedding service returned {} vectors for {} texts",
                    response.vectors.len(),
                    batch_texts.len()
                )));
            }

            for ((index, text), vector) in chunk.iter().zip(response.vectors.into_iter()) {
                outputs[*index] = Some(vector.values.clone());
                fresh_embeddings.push((text.clone(), vector.values));
            }
        }

        if !fresh_embeddings.is_empty() {
            let mut cache = self.cache.lock().await;
            for (text, vector) in &fresh_embeddings {
                cache.insert(text, vector.clone());
            }
        }

        outputs
            .into_iter()
            .map(|vector| {
                vector.ok_or_else(|| {
                    VectorError::Embedding("embedding response did not contain all vectors".into())
                })
            })
            .collect()
    }

    /// Embed a single text.
    #[instrument(skip(self, text))]
    async fn embed_one(&self, text: &str) -> VectorResult<Vec<f32>> {
        self.embed(vec![text.to_string()])
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| {
                VectorError::Embedding(
                    "embedding response was empty for single-text request".into(),
                )
            })
    }

    /// Health-check the embedding service.
    #[instrument(skip(self))]
    async fn health_check(&self) -> VectorResult<bool> {
        let mut client = self.client.clone();
        let mut request = Request::new(HealthRequest {});
        request.set_timeout(Duration::from_millis(self.config.embedding_timeout_ms));
        let response = client
            .health(request)
            .await
            .map_err(VectorError::from)?
            .into_inner();
        Ok(response.ready)
    }

    /// Return metadata about the currently loaded model.
    #[instrument(skip(self))]
    async fn model_info(&self) -> VectorResult<ModelInfo> {
        let mut client = self.client.clone();
        let mut request = Request::new(ModelInfoRequest {});
        request.set_timeout(Duration::from_millis(self.config.embedding_timeout_ms));
        let response = client
            .model_info(request)
            .await
            .map_err(VectorError::from)?
            .into_inner();
        Ok(ModelInfo {
            model_name: response.model_name,
            dimensions: response.dimensions as usize,
            max_sequence_length: response.max_sequence_length as usize,
            device: response.device,
        })
    }

    /// Return cache statistics for the client-local embedding cache.
    async fn cache_stats(&self) -> Option<EmbeddingCacheStats> {
        Some(self.cache.lock().await.stats())
    }
}
