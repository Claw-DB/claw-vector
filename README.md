# claw-vector

claw-vector is the semantic memory engine behind ClawDB. It combines SQLite-backed metadata and FTS5 keyword search, memory-mapped vector persistence, adaptive Flat and HNSW indexing, hybrid retrieval, metadata filtering, reranking, and a Python embedding microservice exposed over gRPC and HTTP.

## What It Provides

- Persistent vector storage with SQLite metadata and mmap-backed raw vectors.
- Automatic index selection: Flat for small collections, HNSW once collections grow.
- ANN, filtered, and hybrid vector plus keyword retrieval.
- Search responses that include both ranked hits and execution metrics.
- A Python embedding service built on sentence-transformers with Prometheus metrics and health probes.
- Rust library and gRPC server entrypoints for application integration.

## Architecture

```text
						   +----------------------------------+
						   | Python embedding service         |
Text ingest and queries -->| sentence-transformers / ONNX     |
						   | gRPC :50051  HTTP :8080          |
						   +----------------+-----------------+
											|
											v
+------------------------+     +------------------------------+
| VectorEngine           |---->| CollectionManager            |
| Rust API and gRPC      |     | lifecycle, persistence,      |
| ANN and hybrid search  |     | index selection, restore     |
+-----------+------------+     +-----------+------------------+
			|                                  |
			v                                  v
 +--------------------------+      +--------------------------+
 | ANN and rerank pipeline  |      | SQLite metadata store    |
 | Flat or HNSW             |      | collections, records,    |
 | filters, hybrid fusion   |      | UUID mapping, FTS5       |
 +-------------+------------+      +-------------+------------+
			   |                                   |
			   v                                   v
	  +------------------+               +---------------------+
	  | mmap vector file |               | persisted indexes   |
	  | raw f32 payloads |               | Flat and HNSW state |
	  +------------------+               +---------------------+
```

## Storage Model

Collection definitions, text payloads, metadata, UUID mappings, and keyword-search state live in SQLite. The database is opened in WAL mode and schema migrations are applied automatically on startup.

Raw vectors are stored separately in fixed-slot memory-mapped files. That keeps metadata queries cheap while letting the engine hydrate vectors only when a request actually asks for them or when reranking needs them.

Each collection starts on a Flat index and automatically switches to HNSW around 1,000 vectors. The collection manager persists the chosen index type so reopen and restore follow the same search path that was active before shutdown.

## Search Model

The Rust API returns a `SearchResponse`:

```rust
pub struct SearchResponse {
	pub results: Vec<SearchResult>,
	pub metrics: SearchMetrics,
}
```

`SearchMetrics` includes query dimensionality, candidate counts, post-filter counts, and end-to-end latency in microseconds.

Metadata filters support:

- `eq`
- `gt`
- `lt`
- `contains`
- `in`
- `exists`
- `and`
- `or`
- `not`

Hybrid retrieval combines ANN candidates with SQLite FTS5 keyword hits. The `alpha` field on `HybridQuery` controls the blend between vector similarity and keyword relevance, where `1.0` is vector-only and `0.0` is keyword-only.

Available rerankers:

- `diversity` for MMR-style result diversification
- `recency` for boosting newer records
- `composite` for chaining reranking passes

## Quick Start

### 1. Start the embedding service

With Docker Compose:

```bash
docker compose up --build claw-vector-svc
```

Or run it locally:

```bash
cd python
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,inference]"
claw-vector-svc
```

The service exposes gRPC on `127.0.0.1:50051` and HTTP on `127.0.0.1:8080` by default.

### 2. Use the Rust engine

```rust
use claw_vector::{
	DistanceMetric, MetadataFilter, SearchQuery, VectorConfig, VectorEngine,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
	let config = VectorConfig::builder()
		.db_path("data/claw_vector.db")
		.index_dir("data/claw_vector_indices")
		.embedding_service_url("http://127.0.0.1:50051")
		.build()?;

	let engine = VectorEngine::new(config).await?;

	engine
		.create_collection("docs", 384, DistanceMetric::Cosine)
		.await?;

	engine
		.upsert(
			"docs",
			"claw-vector persists vector metadata in SQLite",
			json!({"tenant": "acme", "topic": "storage", "priority": 3}),
		)
		.await?;

	let response = engine
		.search(SearchQuery {
			collection: "docs".into(),
			vector: vec![0.12; 384],
			top_k: 5,
			filter: Some(MetadataFilter::Eq {
				key: "tenant".into(),
				value: json!("acme"),
			}),
			include_vectors: false,
			include_metadata: true,
			ef_search: Some(64),
			reranker: None,
		})
		.await?;

	println!("results={}", response.results.len());
	println!("latency_us={}", response.metrics.latency_us);

	engine.close().await?;
	Ok(())
}
```

### 3. Optional: run the Rust gRPC server

```bash
cargo run --bin claw-vector-server
```

The server listens on `0.0.0.0:50051` by default. Override that with `CLAW_GRPC_ADDR`.

## Embedding Service

The Python service loads the embedding model during FastAPI lifespan startup, warms it up, then starts the async gRPC server used by the Rust engine.

HTTP endpoints:

- `GET /health`
- `GET /ready`
- `POST /embed`
- `POST /batch-embed`
- `GET /model-info`
- `GET /metrics`

Authentication:

- HTTP requests accept `X-Claw-Api-Key`.
- gRPC requests accept `x-claw-api-key` (Rust server) or `authorization: Bearer <key>` (Python embedding service).
- When keys are configured, unauthorized requests return `401` (HTTP) or `Unauthenticated` (gRPC).

Example request:

```bash
curl http://127.0.0.1:8080/embed \
  -H 'content-type: application/json' \
  -d '{"texts":["semantic memory for clawdb"],"normalize":true}'
```

Example response shape:

```json
{
  "vectors": [
	{
	  "values": [0.01, -0.03, 0.42],
	  "dimensions": 384
	}
  ],
  "model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "latency_ms": 12
}
```

## Configuration Reference

### Rust engine

`VectorConfig::builder()` controls all runtime settings. The path and embedding endpoint settings can also be supplied through environment variables.

| Setting | Default | Source | Description |
| --- | --- | --- | --- |
| `db_path` | `claw_vector.db` | builder, `CLAW_VECTOR_DB_PATH` | SQLite database file for collections, metadata, and FTS5 state. |
| `index_dir` | `claw_vector_indices` | builder, `CLAW_VECTOR_INDEX_DIR` | Directory for persisted index files and mmap vector files. |
| `embedding_service_url` | `http://localhost:50051` | builder, `CLAW_EMBEDDING_URL` | gRPC endpoint for the Python embedding service. |
| `default_dimensions` | `384` | builder | Default embedding dimensionality. |
| `ef_construction` | `200` | builder | HNSW build-time recall and speed tradeoff. |
| `m_connections` | `16` | builder | HNSW graph degree. |
| `ef_search` | `50` | builder | Default HNSW search breadth. |
| `max_elements` | `1_000_000` | builder | Maximum vectors per collection index. |
| `cache_size` | `10_000` | builder | Rust-side embedding LRU cache capacity. |
| `batch_size` | `64` | builder | Max texts per embedding request. |
| `embedding_timeout_ms` | `5000` | builder | gRPC timeout for embedding calls. |
| `num_threads` | available parallelism | builder | Rayon worker count for index operations. |
| `default_workspace_id` | `default` | builder, `CLAW_DEFAULT_WORKSPACE_ID` | Default tenant/workspace id used when a request does not provide one. |
| `api_key_store_path` | `claw_vector_auth.db` | builder, `CLAW_API_KEY_STORE_PATH` | SQLite database used by Rust gRPC auth key store. |
| `rate_limit_rps` | `100` | builder, `CLAW_RATE_LIMIT_RPS` | Default per-workspace request rate limit for Rust gRPC APIs. |
| `require_auth` | `true` in release, `false` in tests | builder, `CLAW_REQUIRE_AUTH` | Enable or disable auth checks (intended for local development/testing only when false). |
| `CLAW_GRPC_ADDR` | `0.0.0.0:50051` | env | Listen address for the Rust gRPC server binary. |

### Python embedding service

| Environment variable | Default | Description |
| --- | --- | --- |
| `MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face model to load. |
| `DEVICE` | `cpu` | Runtime device such as `cpu`, `cuda`, or another supported backend. |
| `GRPC_HOST` | `0.0.0.0` | gRPC bind host. |
| `GRPC_PORT` | `50051` | gRPC bind port. |
| `HTTP_HOST` | `0.0.0.0` | HTTP bind host. |
| `HTTP_PORT` | `8080` | HTTP bind port. |
| `MAX_BATCH_SIZE` | `64` | Max texts accepted in a single embedding request. |
| `CACHE_SIZE` | `10000` | In-process embedding cache capacity. |
| `NORMALIZE_EMBEDDINGS` | `true` | Whether embeddings are normalized by default. |
| `ONNX_MODEL_PATH` | unset | Optional ONNX model path for alternate inference. |
| `MAX_SEQUENCE_LENGTH` | `256` | Max token length exposed in model metadata responses. |
| `CLAW_API_KEY` | unset | Single API key accepted by Python HTTP/gRPC endpoints. |
| `CLAW_API_KEYS` | unset | Comma-separated list of API keys accepted by Python HTTP/gRPC endpoints. |
| `EMBED_RATE_LIMIT_PER_MINUTE` | `200` | Per-API-key HTTP request limit for `/embed`. |

## Testing And Validation

Rust:

```bash
cargo test --lib --tests
cargo check --benches
```

Python:

```bash
cd python
pytest tests/
```

The engine test suite covers collection lifecycle, persistence across reopen, hybrid search, metadata filters, embedding cache behavior, and Flat to HNSW migration.

## Benchmarks And Performance Targets

Criterion benchmarks live in `benches/vector_bench.rs` and cover:

- single-vector upsert
- batch upsert of 100 records
- Flat search at 500 vectors
- HNSW search at 1,000 and 100,000 vectors
- filtered ANN search at 10,000 vectors
- embedding cache hit latency
- hybrid search at 1,000 text-backed records

Operational targets for the current implementation:

- HNSW search p99 under 10 ms at 100k vectors
- cache-hit embedding lookups under 1 us
- embedding throughput above 2,000 texts per minute for the default MiniLM model on suitable hardware

Use Criterion directly when you want full benchmark runs:

```bash
cargo bench --bench vector_bench
```

## Development Notes

- Rust protobuf generation is handled in `build.rs` with vendored `protoc`, so local protobuf installation is not required.
- SQLite migrations are embedded and applied automatically by the Rust store layer.
- `docker-compose.yml` provides a ready-to-run embedding service and an optional Rust dev container profile.
