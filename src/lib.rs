// lib.rs — crate root: public re-exports and crate-level lint configuration.
#![deny(missing_docs)]

//! `claw-vector` is the semantic memory engine for ClawDB.
//!
//! It provides HNSW-backed approximate nearest-neighbour search, SQLite persistence,
//! memory-mapped vector files, and a gRPC client to a Python embedding microservice.

/// High-level collection management operations.
pub mod collections;
/// Engine configuration and builder.
pub mod config;
/// gRPC client to the Python embedding microservice and related types.
pub mod embeddings;
/// Top-level VectorEngine lifecycle coordinator.
pub mod engine;
/// Unified error type and result alias.
pub mod error;
/// gRPC server and generated proto bindings.
pub mod grpc;
/// HNSW, flat, and selector index implementations.
pub mod index;
/// ANN search, hybrid search, reranking, and metadata filters.
pub mod search;
/// SQLite and mmap vector storage backends.
pub mod store;
/// Core domain types (VectorRecord, Collection, SearchResult, etc.).
pub mod types;

pub use collections::CollectionManager;
pub use config::{VectorConfig, VectorConfigBuilder};
pub use engine::VectorEngine;
pub use error::{VectorError, VectorResult};
pub use types::{
    Collection, DistanceMetric, IndexType, MetadataFilter, SearchQuery, SearchResult,
    VectorRecord,
};
