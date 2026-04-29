// search/rerank.rs — post-retrieval reranking strategies.
use std::cmp::Ordering;

use async_trait::async_trait;
use chrono::Utc;

use crate::{
    error::VectorResult,
    types::{RerankerConfig, SearchResult},
};

/// Post-retrieval reranker interface.
#[async_trait]
pub trait Reranker {
    /// Rerank a list of search results for the provided query vector.
    async fn rerank(
        &self,
        query: &[f32],
        results: Vec<SearchResult>,
    ) -> VectorResult<Vec<SearchResult>>;
}

/// Placeholder cross-encoder reranker with a future gRPC hook.
pub struct CrossEncoderReranker {
    /// Whether a scoring backend is currently available.
    pub service_available: bool,
}

#[async_trait]
impl Reranker for CrossEncoderReranker {
    async fn rerank(
        &self,
        _query: &[f32],
        results: Vec<SearchResult>,
    ) -> VectorResult<Vec<SearchResult>> {
        let _ = self.service_available;
        Ok(results)
    }
}

/// Diversity-promoting reranker based on maximal marginal relevance.
pub struct DiversityReranker {
    /// Relevance-vs-diversity balance in the range `[0.0, 1.0]`.
    pub lambda: f32,
}

#[async_trait]
impl Reranker for DiversityReranker {
    async fn rerank(
        &self,
        query: &[f32],
        results: Vec<SearchResult>,
    ) -> VectorResult<Vec<SearchResult>> {
        Ok(mmr_select(query, &results, self.lambda, results.len()))
    }
}

/// Recency-based reranker.
pub struct RecencyReranker {
    /// Weight applied to the recency boost.
    pub recency_weight: f32,
    /// Exponential decay half-life in days.
    pub half_life_days: f32,
}

#[async_trait]
impl Reranker for RecencyReranker {
    async fn rerank(
        &self,
        _query: &[f32],
        mut results: Vec<SearchResult>,
    ) -> VectorResult<Vec<SearchResult>> {
        let now = Utc::now();
        let half_life_days = self.half_life_days.max(0.001);

        for result in &mut results {
            let age_seconds = now
                .signed_duration_since(result.created_at)
                .num_seconds()
                .max(0) as f32;
            let age_days = age_seconds / 86_400.0;
            let decay_factor = (-age_days / half_life_days).exp();
            result.score *= 1.0 + self.recency_weight * decay_factor;
        }

        sort_results_desc(&mut results);
        Ok(results)
    }
}

/// Apply multiple rerankers in sequence.
pub struct CompositeReranker(pub Vec<Box<dyn Reranker + Send + Sync>>);

#[async_trait]
impl Reranker for CompositeReranker {
    async fn rerank(
        &self,
        query: &[f32],
        mut results: Vec<SearchResult>,
    ) -> VectorResult<Vec<SearchResult>> {
        for reranker in &self.0 {
            results = reranker.rerank(query, results).await?;
        }
        Ok(results)
    }
}

/// Select a diverse subset of candidates using maximal marginal relevance.
pub fn mmr_select(
    query: &[f32],
    candidates: &[SearchResult],
    lambda: f32,
    top_k: usize,
) -> Vec<SearchResult> {
    if candidates.is_empty() || top_k == 0 {
        return Vec::new();
    }

    let lambda = lambda.clamp(0.0, 1.0);
    let mut remaining = candidates.to_vec();
    let mut selected = Vec::with_capacity(top_k.min(candidates.len()));

    while !remaining.is_empty() && selected.len() < top_k {
        let best_index = remaining
            .iter()
            .enumerate()
            .map(|(index, candidate)| {
                let relevance = query_relevance(query, candidate);
                let max_similarity = selected
                    .iter()
                    .map(|selected| candidate_similarity(candidate, selected))
                    .fold(0.0, f32::max);
                let score = lambda * relevance - (1.0 - lambda) * max_similarity;
                (index, score)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
            .map(|(index, _)| index)
            .unwrap_or(0);

        selected.push(remaining.remove(best_index));
    }

    selected
}

/// Apply a configured reranker chain to search results.
pub async fn apply_reranker_config(
    query: &[f32],
    results: Vec<SearchResult>,
    config: Option<&RerankerConfig>,
) -> VectorResult<Vec<SearchResult>> {
    match config {
        None | Some(RerankerConfig::None) => Ok(results),
        Some(config) => build_reranker(config).rerank(query, results).await,
    }
}

/// Return `true` when a configured reranker requires access to raw vectors.
pub fn reranker_needs_vectors(config: Option<&RerankerConfig>) -> bool {
    match config {
        None | Some(RerankerConfig::None) => false,
        Some(RerankerConfig::Diversity { .. }) => true,
        Some(RerankerConfig::Recency { .. }) => false,
        Some(RerankerConfig::Composite(configs)) => configs
            .iter()
            .any(|config| reranker_needs_vectors(Some(config))),
    }
}

fn build_reranker(config: &RerankerConfig) -> Box<dyn Reranker + Send + Sync> {
    match config {
        RerankerConfig::None => Box::new(CompositeReranker(Vec::new())),
        RerankerConfig::Diversity { lambda } => Box::new(DiversityReranker { lambda: *lambda }),
        RerankerConfig::Recency {
            weight,
            half_life_days,
        } => Box::new(RecencyReranker {
            recency_weight: *weight,
            half_life_days: *half_life_days,
        }),
        RerankerConfig::Composite(configs) => Box::new(CompositeReranker(
            configs.iter().map(build_reranker).collect(),
        )),
    }
}

fn query_relevance(query: &[f32], candidate: &SearchResult) -> f32 {
    if let Some(vector) = candidate.vector.as_deref() {
        cosine_similarity(query, vector)
            .map(|similarity| ((similarity + 1.0) / 2.0).clamp(0.0, 1.0))
            .unwrap_or(candidate.score)
    } else {
        candidate.score
    }
}

fn candidate_similarity(left: &SearchResult, right: &SearchResult) -> f32 {
    match (left.vector.as_deref(), right.vector.as_deref()) {
        (Some(left), Some(right)) => cosine_similarity(left, right)
            .map(|similarity| ((similarity + 1.0) / 2.0).clamp(0.0, 1.0))
            .unwrap_or(0.0),
        _ => 0.0,
    }
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> Option<f32> {
    if left.len() != right.len() || left.is_empty() {
        return None;
    }

    let dot: f32 = left.iter().zip(right.iter()).map(|(a, b)| a * b).sum();
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        None
    } else {
        Some(dot / (left_norm * right_norm))
    }
}

fn sort_results_desc(results: &mut [SearchResult]) {
    results.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
    });
}
