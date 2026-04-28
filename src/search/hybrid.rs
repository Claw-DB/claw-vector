// search/hybrid.rs — hybrid ANN + keyword search fusion.
use crate::{
    error::VectorResult,
    types::SearchResult,
};

/// Strategy for combining ANN scores with keyword match scores.
#[derive(Debug, Clone, Copy)]
pub enum HybridStrategy {
    /// Weighted linear combination: `alpha * ann_score + (1 − alpha) * keyword_score`.
    WeightedSum {
        /// Weight given to the ANN score (keyword weight = 1 − alpha).
        alpha: f32,
    },
    /// Reciprocal rank fusion of two sorted result lists.
    ReciprocalRankFusion {
        /// Smoothing constant (default 60).
        k: f32,
    },
}

impl Default for HybridStrategy {
    fn default() -> Self {
        HybridStrategy::ReciprocalRankFusion { k: 60.0 }
    }
}

/// Fuse two ranked result lists (ANN + keyword) using the chosen strategy.
pub fn fuse(
    ann_results: Vec<SearchResult>,
    keyword_results: Vec<SearchResult>,
    strategy: HybridStrategy,
) -> VectorResult<Vec<SearchResult>> {
    match strategy {
        HybridStrategy::WeightedSum { alpha } => weighted_sum(ann_results, keyword_results, alpha),
        HybridStrategy::ReciprocalRankFusion { k } => rrf(ann_results, keyword_results, k),
    }
}

fn weighted_sum(
    ann: Vec<SearchResult>,
    kw: Vec<SearchResult>,
    alpha: f32,
) -> VectorResult<Vec<SearchResult>> {
    use std::collections::HashMap;
    let mut map: HashMap<uuid::Uuid, (SearchResult, f32)> = HashMap::new();
    for (rank, r) in ann.iter().enumerate() {
        let s = alpha * (1.0 / (1.0 + rank as f32));
        map.entry(r.id).or_insert_with(|| (r.clone(), 0.0)).1 += s;
    }
    for (rank, r) in kw.iter().enumerate() {
        let s = (1.0 - alpha) * (1.0 / (1.0 + rank as f32));
        map.entry(r.id).or_insert_with(|| (r.clone(), 0.0)).1 += s;
    }
    let mut results: Vec<SearchResult> = map
        .into_values()
        .map(|(mut r, score)| { r.score = score; r })
        .collect();
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    Ok(results)
}

fn rrf(ann: Vec<SearchResult>, kw: Vec<SearchResult>, k: f32) -> VectorResult<Vec<SearchResult>> {
    use std::collections::HashMap;
    let mut map: HashMap<uuid::Uuid, (SearchResult, f32)> = HashMap::new();
    for (rank, r) in ann.iter().enumerate() {
        let s = 1.0 / (k + rank as f32 + 1.0);
        map.entry(r.id).or_insert_with(|| (r.clone(), 0.0)).1 += s;
    }
    for (rank, r) in kw.iter().enumerate() {
        let s = 1.0 / (k + rank as f32 + 1.0);
        map.entry(r.id).or_insert_with(|| (r.clone(), 0.0)).1 += s;
    }
    let mut results: Vec<SearchResult> = map
        .into_values()
        .map(|(mut r, score)| { r.score = score; r })
        .collect();
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    Ok(results)
}
