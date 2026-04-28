// search/mod.rs — public re-exports for the search sub-module.
/// Approximate nearest-neighbour search orchestration.
pub mod ann;
/// Metadata filter DSL extension methods and helpers.
pub mod filters;
/// Hybrid ANN + keyword search fusion.
pub mod hybrid;
/// Score fusion and reranking strategies.
pub mod rerank;

pub use ann::AnnSearch;
pub use filters::MetadataFilterExt;
pub use rerank::RerankStrategy;
