// search/mod.rs — public re-exports for the search sub-module.
/// Approximate nearest-neighbour search orchestration.
pub mod ann;
/// Metadata filter DSL extension methods and helpers.
pub mod filters;
/// Hybrid ANN + keyword search fusion.
pub mod hybrid;
/// Score fusion and reranking strategies.
pub mod rerank;

pub use ann::AnnSearcher;
pub use filters::{apply_filter, parse_json_path, validate_filter, MetadataFilterExt};
pub use hybrid::HybridSearcher;
pub use rerank::{
    apply_reranker_config, mmr_select, reranker_needs_vectors, CompositeReranker,
    CrossEncoderReranker, DiversityReranker, RecencyReranker, Reranker,
};
