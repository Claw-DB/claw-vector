// search/rerank.rs — score fusion and reranking strategies.
use crate::types::SearchResult;

/// Available reranking strategies.
#[derive(Debug, Clone, Copy)]
pub enum RerankStrategy {
    /// Return results as-is (identity).
    None,
    /// Reciprocal rank fusion with smoothing constant `k` (default 60).
    ReciprocalRankFusion {
        /// Smoothing constant.
        k: f32,
    },
    /// Cross-encoder placeholder: re-scores each result with a provided closure.
    CrossEncoder,
}

impl Default for RerankStrategy {
    fn default() -> Self {
        RerankStrategy::None
    }
}

/// Apply a reranking strategy to a list of results.
pub fn rerank(mut results: Vec<SearchResult>, strategy: RerankStrategy) -> Vec<SearchResult> {
    match strategy {
        RerankStrategy::None => results,
        RerankStrategy::ReciprocalRankFusion { k } => {
            for (rank, r) in results.iter_mut().enumerate() {
                r.score = 1.0 / (k + rank as f32 + 1.0);
            }
            results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            results
        }
        RerankStrategy::CrossEncoder => results, // stub — real impl plugs in a cross-encoder model
    }
}
