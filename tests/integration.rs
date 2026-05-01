use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
};

use async_trait::async_trait;
use serde_json::json;
use sha2::{Digest, Sha256};
use tempfile::TempDir;

use claw_vector::{
    embeddings::{EmbeddingCacheStats, EmbeddingProvider, ModelInfo},
    DistanceMetric, HybridQuery, MetadataFilter, SearchQuery, VectorConfig, VectorEngine,
    VectorError, VectorResult,
};

struct MockEmbeddingClient {
    dimensions: usize,
    cache: tokio::sync::Mutex<HashMap<String, Vec<f32>>>,
    hit_count: AtomicU64,
    miss_count: AtomicU64,
    embed_calls: AtomicUsize,
}

impl MockEmbeddingClient {
    fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            cache: tokio::sync::Mutex::new(HashMap::new()),
            hit_count: AtomicU64::new(0),
            miss_count: AtomicU64::new(0),
            embed_calls: AtomicUsize::new(0),
        }
    }

    fn deterministic_vector(&self, text: &str) -> Vec<f32> {
        let digest = Sha256::digest(text.as_bytes());
        let mut vector = Vec::with_capacity(self.dimensions);
        let mut counter = 0u64;
        while vector.len() < self.dimensions {
            let mut hasher = Sha256::new();
            hasher.update(digest);
            hasher.update(counter.to_le_bytes());
            let block = hasher.finalize();
            for chunk in block.chunks(4) {
                if vector.len() == self.dimensions {
                    break;
                }
                let mut bytes = [0u8; 4];
                bytes.copy_from_slice(chunk);
                let value = u32::from_le_bytes(bytes);
                vector.push((value as f32 / u32::MAX as f32) * 2.0 - 1.0);
            }
            counter += 1;
        }
        vector
    }

    fn embed_call_count(&self) -> usize {
        self.embed_calls.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl EmbeddingProvider for MockEmbeddingClient {
    async fn embed(&self, texts: Vec<String>) -> VectorResult<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        let mut cache = self.cache.lock().await;
        for text in texts {
            if let Some(vector) = cache.get(&text) {
                self.hit_count.fetch_add(1, Ordering::Relaxed);
                results.push(vector.clone());
            } else {
                self.miss_count.fetch_add(1, Ordering::Relaxed);
                self.embed_calls.fetch_add(1, Ordering::Relaxed);
                let vector = self.deterministic_vector(&text);
                cache.insert(text.clone(), vector.clone());
                results.push(vector);
            }
        }
        Ok(results)
    }

    async fn embed_one(&self, text: &str) -> VectorResult<Vec<f32>> {
        self.embed(vec![text.to_string()])
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| VectorError::Embedding("mock embedder returned no vectors".into()))
    }

    async fn health_check(&self) -> VectorResult<bool> {
        Ok(true)
    }

    async fn model_info(&self) -> VectorResult<ModelInfo> {
        Ok(ModelInfo {
            model_name: "mock-embedder".into(),
            dimensions: self.dimensions,
            max_sequence_length: 512,
            device: "cpu".into(),
        })
    }

    async fn cache_stats(&self) -> Option<EmbeddingCacheStats> {
        Some(EmbeddingCacheStats {
            hit_count: self.hit_count.load(Ordering::Relaxed),
            miss_count: self.miss_count.load(Ordering::Relaxed),
            len: self.cache.lock().await.len(),
        })
    }
}

async fn test_engine(
    tempdir: &TempDir,
    dimensions: usize,
    provider: Arc<MockEmbeddingClient>,
) -> VectorEngine {
    let config = VectorConfig::builder()
        .db_path(tempdir.path().join("claw-vector.db"))
        .index_dir(tempdir.path().join("indices"))
        .default_dimensions(dimensions)
        .max_elements(200_000)
        .build()
        .unwrap();
    VectorEngine::with_embedding_provider(config, provider)
        .await
        .unwrap()
}

#[tokio::test]
async fn engine_opens_and_closes() {
    let tempdir = TempDir::new().unwrap();
    let provider = Arc::new(MockEmbeddingClient::new(16));
    let engine = test_engine(&tempdir, 16, provider).await;
    engine.close().await.unwrap();
}

#[tokio::test]
async fn create_and_list_collections() {
    let tempdir = TempDir::new().unwrap();
    let provider = Arc::new(MockEmbeddingClient::new(16));
    let engine = test_engine(&tempdir, 16, provider).await;

    engine
        .create_collection("alpha", 16, DistanceMetric::Cosine)
        .await
        .unwrap();
    engine
        .create_collection("beta", 16, DistanceMetric::Cosine)
        .await
        .unwrap();
    engine
        .create_collection("gamma", 16, DistanceMetric::Cosine)
        .await
        .unwrap();

    let collections = engine.list_collections().await.unwrap();
    let names = collections
        .into_iter()
        .map(|collection| collection.name)
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["alpha", "beta", "gamma"]);
}

#[tokio::test]
async fn delete_collection() {
    let tempdir = TempDir::new().unwrap();
    let provider = Arc::new(MockEmbeddingClient::new(16));
    let engine = test_engine(&tempdir, 16, provider).await;

    engine
        .create_collection("docs", 16, DistanceMetric::Cosine)
        .await
        .unwrap();
    engine.delete_collection("docs").await.unwrap();

    assert!(engine.list_collections().await.unwrap().is_empty());
}

#[tokio::test]
async fn upsert_and_get_vector() {
    let tempdir = TempDir::new().unwrap();
    let provider = Arc::new(MockEmbeddingClient::new(16));
    let engine = test_engine(&tempdir, 16, provider).await;
    engine
        .create_collection("docs", 16, DistanceMetric::Cosine)
        .await
        .unwrap();

    let id = engine
        .upsert("docs", "hello world", json!({"topic": "intro"}))
        .await
        .unwrap();
    let record = engine.get("docs", id).await.unwrap();

    assert_eq!(record.metadata["topic"], json!("intro"));
    assert_eq!(record.text.as_deref(), Some("hello world"));
    assert_eq!(record.vector.len(), 16);
}

#[tokio::test]
async fn search_returns_correct_top_k() {
    let tempdir = TempDir::new().unwrap();
    let provider = Arc::new(MockEmbeddingClient::new(16));
    let engine = test_engine(&tempdir, 16, provider.clone()).await;
    engine
        .create_collection("docs", 16, DistanceMetric::Cosine)
        .await
        .unwrap();

    for index in 0..100 {
        engine
            .upsert(
                "docs",
                &format!("document {index}"),
                json!({"bucket": index % 10}),
            )
            .await
            .unwrap();
    }

    let response = engine
        .search(SearchQuery {
            collection: "docs".into(),
            vector: provider.deterministic_vector("document 0"),
            top_k: 5,
            filter: None,
            include_vectors: false,
            include_metadata: true,
            ef_search: None,
            reranker: None,
        })
        .await
        .unwrap();

    assert_eq!(response.results.len(), 5);
}

#[tokio::test]
async fn search_score_ordering() {
    let tempdir = TempDir::new().unwrap();
    let provider = Arc::new(MockEmbeddingClient::new(16));
    let engine = test_engine(&tempdir, 16, provider.clone()).await;
    engine
        .create_collection("docs", 16, DistanceMetric::Cosine)
        .await
        .unwrap();

    for index in 0..25 {
        engine
            .upsert("docs", &format!("ordered {index}"), json!({"value": index}))
            .await
            .unwrap();
    }

    let response = engine
        .search(SearchQuery {
            collection: "docs".into(),
            vector: provider.deterministic_vector("ordered 0"),
            top_k: 10,
            filter: None,
            include_vectors: false,
            include_metadata: false,
            ef_search: None,
            reranker: None,
        })
        .await
        .unwrap();

    assert!(response
        .results
        .windows(2)
        .all(|pair| pair[0].score >= pair[1].score));
}

#[tokio::test]
async fn search_with_eq_filter() {
    let tempdir = TempDir::new().unwrap();
    let provider = Arc::new(MockEmbeddingClient::new(12));
    let engine = test_engine(&tempdir, 12, provider.clone()).await;
    engine
        .create_collection("docs", 12, DistanceMetric::Cosine)
        .await
        .unwrap();

    for index in 0..20 {
        engine
            .upsert(
                "docs",
                &format!("eq-filter {index}"),
                json!({"tenant": if index % 2 == 0 { "a" } else { "b" }}),
            )
            .await
            .unwrap();
    }

    let response = engine
        .search(SearchQuery {
            collection: "docs".into(),
            vector: provider.deterministic_vector("eq-filter 0"),
            top_k: 10,
            filter: Some(MetadataFilter::Eq {
                key: "tenant".into(),
                value: json!("a"),
            }),
            include_vectors: false,
            include_metadata: true,
            ef_search: None,
            reranker: None,
        })
        .await
        .unwrap();

    assert!(response
        .results
        .iter()
        .all(|result| result.metadata["tenant"] == json!("a")));
}

#[tokio::test]
async fn search_with_and_filter() {
    let tempdir = TempDir::new().unwrap();
    let provider = Arc::new(MockEmbeddingClient::new(12));
    let engine = test_engine(&tempdir, 12, provider.clone()).await;
    engine
        .create_collection("docs", 12, DistanceMetric::Cosine)
        .await
        .unwrap();

    for index in 0..20 {
        engine
            .upsert(
                "docs",
                &format!("and-filter {index}"),
                json!({"tenant": "a", "bucket": index % 3}),
            )
            .await
            .unwrap();
    }

    let response = engine
        .search(SearchQuery {
            collection: "docs".into(),
            vector: provider.deterministic_vector("and-filter 1"),
            top_k: 10,
            filter: Some(MetadataFilter::And(vec![
                MetadataFilter::Eq {
                    key: "tenant".into(),
                    value: json!("a"),
                },
                MetadataFilter::Eq {
                    key: "bucket".into(),
                    value: json!(1),
                },
            ])),
            include_vectors: false,
            include_metadata: true,
            ef_search: None,
            reranker: None,
        })
        .await
        .unwrap();

    assert!(response.results.iter().all(|result| {
        result.metadata["tenant"] == json!("a") && result.metadata["bucket"] == json!(1)
    }));
}

#[tokio::test]
async fn search_with_not_filter() {
    let tempdir = TempDir::new().unwrap();
    let provider = Arc::new(MockEmbeddingClient::new(12));
    let engine = test_engine(&tempdir, 12, provider.clone()).await;
    engine
        .create_collection("docs", 12, DistanceMetric::Cosine)
        .await
        .unwrap();

    for index in 0..20 {
        engine
            .upsert(
                "docs",
                &format!("not-filter {index}"),
                json!({"tenant": if index % 3 == 0 { "blocked" } else { "allowed" }}),
            )
            .await
            .unwrap();
    }

    let response = engine
        .search(SearchQuery {
            collection: "docs".into(),
            vector: provider.deterministic_vector("not-filter 0"),
            top_k: 10,
            filter: Some(MetadataFilter::Not(Box::new(MetadataFilter::Eq {
                key: "tenant".into(),
                value: json!("blocked"),
            }))),
            include_vectors: false,
            include_metadata: true,
            ef_search: None,
            reranker: None,
        })
        .await
        .unwrap();

    assert!(response
        .results
        .iter()
        .all(|result| result.metadata["tenant"] != json!("blocked")));
}

#[tokio::test]
async fn flat_to_hnsw_migration() {
    let tempdir = TempDir::new().unwrap();
    let provider = Arc::new(MockEmbeddingClient::new(8));
    let engine = test_engine(&tempdir, 8, provider).await;
    engine
        .create_collection("docs", 8, DistanceMetric::Cosine)
        .await
        .unwrap();

    for index in 0..1001 {
        engine
            .upsert("docs", &format!("migration {index}"), json!({"n": index}))
            .await
            .unwrap();
    }

    let collection = engine
        .collections
        .get_collection(&engine.config.default_workspace_id, "docs")
        .await
        .unwrap();
    assert_eq!(collection.index_type, claw_vector::IndexType::HNSW);
}

#[tokio::test]
async fn batch_upsert_and_batch_search() {
    let tempdir = TempDir::new().unwrap();
    let provider = Arc::new(MockEmbeddingClient::new(10));
    let engine = test_engine(&tempdir, 10, provider.clone()).await;
    engine
        .create_collection("docs", 10, DistanceMetric::Cosine)
        .await
        .unwrap();

    let items = (0..500)
        .map(|index| (format!("batch {index}"), json!({"group": index % 5})))
        .collect::<Vec<_>>();
    engine.upsert_batch("docs", items).await.unwrap();

    let queries = (0..10)
        .map(|index| SearchQuery {
            collection: "docs".into(),
            vector: provider.deterministic_vector(&format!("batch {index}")),
            top_k: 5,
            filter: None,
            include_vectors: false,
            include_metadata: true,
            ef_search: Some(64),
            reranker: None,
        })
        .collect::<Vec<_>>();

    let responses = engine.ann_searcher.batch_search(queries).await.unwrap();
    assert_eq!(responses.len(), 10);
    assert!(responses
        .iter()
        .all(|response| !response.results.is_empty()));
}

#[tokio::test]
async fn hybrid_search_text_and_vector() {
    let tempdir = TempDir::new().unwrap();
    let provider = Arc::new(MockEmbeddingClient::new(16));
    let engine = test_engine(&tempdir, 16, provider.clone()).await;
    engine
        .create_collection("docs", 16, DistanceMetric::Cosine)
        .await
        .unwrap();

    for index in 0..50 {
        engine
            .upsert(
                "docs",
                &format!("hybrid document {index}"),
                json!({"group": index % 4}),
            )
            .await
            .unwrap();
    }

    let response = engine
        .hybrid_search(HybridQuery {
            collection: "docs".into(),
            vector: provider.deterministic_vector("hybrid document 1"),
            text: Some("document".into()),
            top_k: 5,
            alpha: 0.7,
            filter: None,
            include_vectors: false,
            reranker: None,
        })
        .await
        .unwrap();

    assert_eq!(response.results.len(), 5);
    assert!(response.results.iter().any(|result| {
        result
            .text
            .as_deref()
            .unwrap_or_default()
            .contains("document")
    }));
}

#[tokio::test]
async fn embedding_cache_hit() {
    let tempdir = TempDir::new().unwrap();
    let provider = Arc::new(MockEmbeddingClient::new(16));
    let engine = test_engine(&tempdir, 16, provider.clone()).await;
    engine
        .create_collection("docs", 16, DistanceMetric::Cosine)
        .await
        .unwrap();

    engine
        .upsert("docs", "repeat me", json!({"kind": "a"}))
        .await
        .unwrap();
    engine
        .upsert("docs", "repeat me", json!({"kind": "b"}))
        .await
        .unwrap();

    assert_eq!(provider.embed_call_count(), 1);
}

#[tokio::test]
async fn collection_dimension_mismatch() {
    let tempdir = TempDir::new().unwrap();
    let provider = Arc::new(MockEmbeddingClient::new(16));
    let engine = test_engine(&tempdir, 16, provider).await;
    engine
        .create_collection("docs", 16, DistanceMetric::Cosine)
        .await
        .unwrap();

    let err = engine
        .upsert_vector("docs", vec![0.1; 8], json!({}))
        .await
        .unwrap_err();
    assert!(matches!(err, VectorError::DimensionMismatch { .. }));
}

#[tokio::test]
async fn persistence_across_reopen() {
    let tempdir = TempDir::new().unwrap();
    let provider = Arc::new(MockEmbeddingClient::new(16));
    let engine = test_engine(&tempdir, 16, provider.clone()).await;
    engine
        .create_collection("docs", 16, DistanceMetric::Cosine)
        .await
        .unwrap();

    engine
        .upsert("docs", "persistent record", json!({"persist": true}))
        .await
        .unwrap();
    engine.close().await.unwrap();

    let reopened_provider = Arc::new(MockEmbeddingClient::new(16));
    let reopened = test_engine(&tempdir, 16, reopened_provider.clone()).await;
    let response = reopened
        .search(SearchQuery {
            collection: "docs".into(),
            vector: reopened_provider.deterministic_vector("persistent record"),
            top_k: 3,
            filter: Some(MetadataFilter::Eq {
                key: "persist".into(),
                value: json!(true),
            }),
            include_vectors: false,
            include_metadata: true,
            ef_search: None,
            reranker: None,
        })
        .await
        .unwrap();

    assert!(!response.results.is_empty());
    assert_eq!(response.results[0].metadata["persist"], json!(true));
}
