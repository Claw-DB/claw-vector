// index/mod.rs — public re-exports for the index sub-module.
/// Brute-force flat vector index for small collections.
pub mod flat;
/// HNSW index wrapper around hnsw_rs.
pub mod hnsw;
/// Auto-selecting index that migrates from flat to HNSW at the configured threshold.
pub mod selector;

pub use flat::FlatIndex;
pub use hnsw::{HnswIndex, HnswStats};
pub use selector::{IndexSelector, HNSW_THRESHOLD};
