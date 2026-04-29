// engine.rs — VectorEngine: public entry point that unifies all subsystems.
use std::sync::Arc;

use tracing::instrument;

use crate::{
    collections::CollectionManager,
    config::VectorConfig,
    embeddings::{EmbeddingClient, EmbeddingProvider},
    error::VectorResult,
    search::{AnnSearcher, HybridSearcher},
    store::VectorStore,
    types::{
        Collection, DistanceMetric, EngineStats, HybridQuery, SearchQuery, SearchResponse,
        VectorRecord,
    },
};

/// High-level engine that manages collections, search, and embeddings.
pub struct VectorEngine {
    /// Runtime configuration.
    pub config: VectorConfig,
    /// Collection lifecycle and persistence manager.
    pub collections: Arc<CollectionManager>,
    /// ANN search service.
    pub ann_searcher: Arc<AnnSearcher>,
    /// Hybrid vector + keyword search service.
    pub hybrid_searcher: Arc<HybridSearcher>,
    /// Embedding provider used for text ingestion and search.
    pub embedding_client: Arc<dyn EmbeddingProvider>,
}

impl VectorEngine {
    /// Create a new engine using the configured gRPC embedding service.
    #[instrument]
    pub async fn new(config: VectorConfig) -> VectorResult<Self> {
        let embedding_client =
            Arc::new(EmbeddingClient::new(&config).await?) as Arc<dyn EmbeddingProvider>;
        Self::with_embedding_provider(config, embedding_client).await
    }

    /// Open an engine using the default configuration.
    #[instrument]
    pub async fn open_default() -> VectorResult<Self> {
        Self::new(VectorConfig::default()).await
    }

    /// Create a new engine with a caller-supplied embedding provider.
    #[instrument(skip(embedding_client))]
    pub async fn with_embedding_provider(
        config: VectorConfig,
        embedding_client: Arc<dyn EmbeddingProvider>,
    ) -> VectorResult<Self> {
        let store = Arc::new(VectorStore::new(&config.db_path).await?);
        let collections =
            Arc::new(CollectionManager::new(config.clone(), Arc::clone(&store)).await?);
        let ann_searcher = Arc::new(AnnSearcher::new(Arc::clone(&collections)));
        let hybrid_searcher = Arc::new(HybridSearcher::new(
            Arc::clone(&ann_searcher),
            Arc::clone(&store),
        ));

        Ok(VectorEngine {
            config,
            collections,
            ann_searcher,
            hybrid_searcher,
            embedding_client,
        })
    }

    /// Create a collection with the provided dimensions and distance metric.
    #[instrument(skip(self))]
    pub async fn create_collection(
        &self,
        name: &str,
        dimensions: usize,
        distance: DistanceMetric,
    ) -> VectorResult<Collection> {
        self.collections
            .create_collection(name, dimensions, distance)
            .await
    }

    /// Delete a collection and all of its persisted state.
    #[instrument(skip(self))]
    pub async fn delete_collection(&self, name: &str) -> VectorResult<()> {
        self.collections.delete_collection(name).await
    }

    /// List all collections.
    #[instrument(skip(self))]
    pub async fn list_collections(&self) -> VectorResult<Vec<Collection>> {
        self.collections.list_collections().await
    }

    /// Embed text, persist the record, and return its UUID.
    #[instrument(skip(self, text, metadata))]
    pub async fn upsert(
        &self,
        collection: &str,
        text: &str,
        metadata: serde_json::Value,
    ) -> VectorResult<uuid::Uuid> {
        let vector = self.embedding_client.embed_one(text).await?;
        let record = VectorRecord::new(collection, vector)
            .with_text(text.to_string())
            .with_metadata(metadata);
        self.collections.insert_vector(record).await
    }

    /// Embed and insert multiple text records.
    #[instrument(skip(self, items))]
    pub async fn upsert_batch(
        &self,
        collection: &str,
        items: Vec<(String, serde_json::Value)>,
    ) -> VectorResult<Vec<uuid::Uuid>> {
        let texts = items
            .iter()
            .map(|(text, _)| text.clone())
            .collect::<Vec<_>>();
        let embeddings = self.embedding_client.embed(texts).await?;
        let records = items
            .into_iter()
            .zip(embeddings.into_iter())
            .map(|((text, metadata), vector)| {
                VectorRecord::new(collection, vector)
                    .with_text(text)
                    .with_metadata(metadata)
            })
            .collect::<Vec<_>>();
        self.collections.insert_batch(records).await
    }

    /// Insert a raw vector directly.
    #[instrument(skip(self, vector, metadata))]
    pub async fn upsert_vector(
        &self,
        collection: &str,
        vector: Vec<f32>,
        metadata: serde_json::Value,
    ) -> VectorResult<uuid::Uuid> {
        let record = VectorRecord::new(collection, vector).with_metadata(metadata);
        self.collections.insert_vector(record).await
    }

    /// Execute ANN search.
    #[instrument(skip(self, query))]
    pub async fn search(&self, query: SearchQuery) -> VectorResult<SearchResponse> {
        self.ann_searcher.search(query).await
    }

    /// Execute ANN search from raw text.
    #[instrument(skip(self, text))]
    pub async fn search_text(
        &self,
        collection: &str,
        text: &str,
        top_k: usize,
    ) -> VectorResult<SearchResponse> {
        let vector = self.embedding_client.embed_one(text).await?;
        self.ann_searcher
            .search(SearchQuery {
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

    /// Execute hybrid search.
    #[instrument(skip(self, query))]
    pub async fn hybrid_search(&self, query: HybridQuery) -> VectorResult<SearchResponse> {
        self.hybrid_searcher.search(query).await
    }

    /// Delete a vector record by UUID.
    #[instrument(skip(self))]
    pub async fn delete(&self, collection: &str, id: uuid::Uuid) -> VectorResult<bool> {
        self.collections.delete_vector(collection, id).await
    }

    /// Fetch a vector record by UUID.
    #[instrument(skip(self))]
    pub async fn get(&self, collection: &str, id: uuid::Uuid) -> VectorResult<VectorRecord> {
        self.collections.get_vector(collection, id).await
    }

    /// Persist indexes and close the underlying store.
    #[instrument(skip(self))]
    pub async fn close(&self) -> VectorResult<()> {
        self.collections.persist_indexes().await?;
        self.collections.store.close().await;
        Ok(())
    }

    /// Return runtime statistics for the engine.
    #[instrument(skip(self))]
    pub async fn stats(&self) -> EngineStats {
        let collections = self
            .collections
            .list_collections()
            .await
            .unwrap_or_default();
        let cache_stats = self
            .embedding_client
            .cache_stats()
            .await
            .unwrap_or_default();

        EngineStats {
            collection_count: collections.len(),
            total_vectors: collections
                .iter()
                .map(|collection| collection.vector_count)
                .sum(),
            loaded_indexes: self.collections.loaded_index_count().await,
            loaded_mmap_files: self.collections.loaded_mmap_count().await,
            embedding_cache_hits: cache_stats.hit_count,
            embedding_cache_misses: cache_stats.miss_count,
        }
    }
}
