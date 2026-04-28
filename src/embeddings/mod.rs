// embeddings/mod.rs — public re-exports for the embeddings sub-module.
/// LRU cache for embedding vectors keyed by text.
pub mod cache;
/// gRPC client to the Python embedding service.
pub mod client;
/// Domain types for embedding requests and responses.
pub mod types;

pub use cache::EmbeddingCache;
pub use client::EmbeddingClient;
pub use types::{EmbeddingRequest, EmbeddingResponse, EmbedVector};
