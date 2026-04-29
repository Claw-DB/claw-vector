use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use serde_json::json;
use sha2::{Digest, Sha256};
use tempfile::TempDir;
use tokio::runtime::Runtime;

use claw_vector::{
    embeddings::{EmbeddingCacheStats, EmbeddingProvider, ModelInfo},
    DistanceMetric, HybridQuery, SearchQuery, VectorConfig, VectorEngine, VectorResult,
};

struct BenchHarness {
    _tempdir: TempDir,
    engine: Arc<VectorEngine>,
}

struct MockEmbeddingProvider {
    dimensions: usize,
}

impl MockEmbeddingProvider {
    fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }

    fn vector_for(&self, text: &str) -> Vec<f32> {
        let digest = Sha256::digest(text.as_bytes());
        let mut vector = Vec::with_capacity(self.dimensions);
        while vector.len() < self.dimensions {
            for chunk in digest.chunks(4) {
                if vector.len() == self.dimensions {
                    break;
                }
                let mut bytes = [0u8; 4];
                bytes[..chunk.len()].copy_from_slice(chunk);
                let raw = u32::from_le_bytes(bytes);
                vector.push((raw as f32 / u32::MAX as f32) * 2.0 - 1.0);
            }
        }
        vector
    }
}

#[async_trait]
impl EmbeddingProvider for MockEmbeddingProvider {
    async fn embed(&self, texts: Vec<String>) -> VectorResult<Vec<Vec<f32>>> {
        Ok(texts
            .into_iter()
            .map(|text| self.vector_for(&text))
            .collect())
    }

    async fn embed_one(&self, text: &str) -> VectorResult<Vec<f32>> {
        Ok(self.vector_for(text))
    }

    async fn health_check(&self) -> VectorResult<bool> {
        Ok(true)
    }

    async fn model_info(&self) -> VectorResult<ModelInfo> {
        Ok(ModelInfo {
            model_name: "mock-bench".into(),
            dimensions: self.dimensions,
            max_sequence_length: 512,
            device: "cpu".into(),
        })
    }

    async fn cache_stats(&self) -> Option<EmbeddingCacheStats> {
        Some(EmbeddingCacheStats::default())
    }
}

fn runtime() -> Runtime {
    Runtime::new().expect("failed to create Tokio runtime for benchmarks")
}

async fn create_harness(dimensions: usize) -> BenchHarness {
    let tempdir = TempDir::new().expect("failed to create tempdir for benchmark harness");
    let config = VectorConfig::builder()
        .db_path(tempdir.path().join("bench.db"))
        .index_dir(tempdir.path().join("index"))
        .default_dimensions(dimensions)
        .build()
        .expect("failed to build benchmark config");
    let provider = Arc::new(MockEmbeddingProvider::new(dimensions));
    let engine = Arc::new(
        VectorEngine::with_embedding_provider(config, provider)
            .await
            .expect("failed to create benchmark engine"),
    );
    engine
        .create_collection("bench", dimensions, DistanceMetric::Cosine)
        .await
        .expect("failed to create benchmark collection");
    BenchHarness {
        _tempdir: tempdir,
        engine,
    }
}

fn create_harness_blocking(runtime: &Runtime, dimensions: usize) -> BenchHarness {
    runtime.block_on(create_harness(dimensions))
}

fn create_seeded_harness(runtime: &Runtime, dimensions: usize, count: usize) -> BenchHarness {
    runtime.block_on(async {
        let harness = create_harness(dimensions).await;
        seed_vectors(&harness.engine, count, dimensions).await;
        harness
    })
}

async fn seed_vectors(engine: &VectorEngine, count: usize, dimensions: usize) {
    let mut batch = Vec::with_capacity(count);
    for index in 0..count {
        let vector = (0..dimensions)
            .map(|dim| ((index + dim) % 97) as f32 / 97.0)
            .collect::<Vec<_>>();
        batch.push((
            format!("document {index}"),
            json!({
                "bucket": index % 10,
                "tag": format!("tag-{}", index % 7),
            }),
            vector,
        ));
    }

    for (text, metadata, vector) in batch {
        engine
            .upsert_vector("bench", vector, metadata)
            .await
            .expect("failed to seed benchmark vector");
        let _ = text;
    }
}

fn bench_upsert_single(c: &mut Criterion) {
    let runtime = runtime();
    c.bench_function("bench_upsert_single", |b| {
        b.iter_batched(
            || create_harness_blocking(&runtime, 32),
            |harness| {
                runtime.block_on(async move {
                    harness
                        .engine
                        .upsert_vector("bench", vec![0.25; 32], json!({"kind": "single"}))
                        .await
                        .expect("single upsert failed");
                });
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_upsert_batch_100(c: &mut Criterion) {
    let runtime = runtime();
    c.bench_function("bench_upsert_batch_100", |b| {
        b.iter_batched(
            || create_harness_blocking(&runtime, 32),
            |harness| {
                runtime.block_on(async move {
                    let items = (0..100)
                        .map(|index| (format!("batch text {index}"), json!({"group": index % 5})))
                        .collect::<Vec<_>>();
                    harness
                        .engine
                        .upsert_batch("bench", items)
                        .await
                        .expect("batch upsert failed");
                });
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_search_hnsw_1k(c: &mut Criterion) {
    let runtime = runtime();
    c.bench_function("bench_search_hnsw_1k", |b| {
        b.iter_batched(
            || create_seeded_harness(&runtime, 32, 1_000),
            |harness| {
                runtime.block_on(async move {
                    harness
                        .engine
                        .search(SearchQuery {
                            collection: "bench".into(),
                            vector: vec![0.3; 32],
                            top_k: 10,
                            filter: None,
                            include_vectors: false,
                            include_metadata: true,
                            ef_search: Some(64),
                            reranker: None,
                        })
                        .await
                        .expect("hnsw 1k search failed");
                });
            },
            BatchSize::LargeInput,
        );
    });
}

fn bench_search_hnsw_100k(c: &mut Criterion) {
    let runtime = runtime();
    let mut group = c.benchmark_group("bench_search_hnsw_100k");
    group.measurement_time(Duration::from_secs(10));
    group.bench_function(BenchmarkId::new("search", 100_000), |b| {
        b.iter_batched(
            || create_seeded_harness(&runtime, 16, 100_000),
            |harness| {
                runtime.block_on(async move {
                    // Target: HNSW search p99 < 10ms at 100k vectors.
                    harness
                        .engine
                        .search(SearchQuery {
                            collection: "bench".into(),
                            vector: vec![0.2; 16],
                            top_k: 10,
                            filter: None,
                            include_vectors: false,
                            include_metadata: false,
                            ef_search: Some(96),
                            reranker: None,
                        })
                        .await
                        .expect("hnsw 100k search failed");
                });
            },
            BatchSize::PerIteration,
        );
    });
    group.finish();
}

fn bench_flat_search_500(c: &mut Criterion) {
    let runtime = runtime();
    c.bench_function("bench_flat_search_500", |b| {
        b.iter_batched(
            || create_seeded_harness(&runtime, 32, 500),
            |harness| {
                runtime.block_on(async move {
                    harness
                        .engine
                        .search(SearchQuery {
                            collection: "bench".into(),
                            vector: vec![0.1; 32],
                            top_k: 10,
                            filter: None,
                            include_vectors: false,
                            include_metadata: false,
                            ef_search: None,
                            reranker: None,
                        })
                        .await
                        .expect("flat search failed");
                });
            },
            BatchSize::LargeInput,
        );
    });
}

fn bench_filter_and_search_10k(c: &mut Criterion) {
    let runtime = runtime();
    c.bench_function("bench_filter_and_search_10k", |b| {
        b.iter_batched(
            || create_seeded_harness(&runtime, 24, 10_000),
            |harness| {
                runtime.block_on(async move {
                    harness
                        .engine
                        .search(SearchQuery {
                            collection: "bench".into(),
                            vector: vec![0.5; 24],
                            top_k: 25,
                            filter: Some(claw_vector::MetadataFilter::Eq {
                                key: "bucket".into(),
                                value: json!(3),
                            }),
                            include_vectors: false,
                            include_metadata: true,
                            ef_search: Some(64),
                            reranker: None,
                        })
                        .await
                        .expect("filtered search failed");
                });
            },
            BatchSize::LargeInput,
        );
    });
}

fn bench_embedding_cache_hit(c: &mut Criterion) {
    let runtime = runtime();
    c.bench_function("bench_embedding_cache_hit", |b| {
        b.iter_batched(
            || create_harness_blocking(&runtime, 32),
            |harness| {
                runtime.block_on(async move {
                    let _ = harness.engine.embedding_client.embed_one("cache-me").await;
                    // Target: cache hit latency < 1us.
                    let _ = harness.engine.embedding_client.embed_one("cache-me").await;
                });
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_hybrid_search_1k(c: &mut Criterion) {
    let runtime = runtime();
    c.bench_function("bench_hybrid_search_1k", |b| {
        b.iter_batched(
            || {
                runtime.block_on(async {
                    let harness = create_harness(32).await;
                    for index in 0..1_000 {
                        harness
                            .engine
                            .upsert(
                                "bench",
                                &format!("hybrid document number {index}"),
                                json!({"bucket": index % 10}),
                            )
                            .await
                            .expect("hybrid upsert failed");
                    }
                    harness
                })
            },
            |harness| {
                runtime.block_on(async move {
                    harness
                        .engine
                        .hybrid_search(HybridQuery {
                            collection: "bench".into(),
                            vector: vec![0.4; 32],
                            text: Some("document".into()),
                            top_k: 10,
                            alpha: 0.7,
                            filter: None,
                            include_vectors: false,
                            reranker: None,
                        })
                        .await
                        .expect("hybrid search failed");
                });
            },
            BatchSize::LargeInput,
        );
    });
}

criterion_group!(
    benches,
    bench_upsert_single,
    bench_upsert_batch_100,
    bench_search_hnsw_1k,
    bench_search_hnsw_100k,
    bench_flat_search_500,
    bench_filter_and_search_10k,
    bench_embedding_cache_hit,
    bench_hybrid_search_1k,
);
criterion_main!(benches);
