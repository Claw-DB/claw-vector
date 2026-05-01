// search/rerank.rs — post-retrieval reranking strategies.
use std::cmp::Ordering;
use std::collections::HashMap;

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
        Some(RerankerConfig::Composite(configs)) => {
            apply_composite_reranker(query, results, configs).await
        }
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
        RerankerConfig::Diversity { lambda, .. } => Box::new(DiversityReranker { lambda: *lambda }),
        RerankerConfig::Recency {
            boost,
            half_life_days,
            ..
        } => Box::new(RecencyReranker {
            recency_weight: *boost,
            half_life_days: *half_life_days,
        }),
        RerankerConfig::Composite(configs) => Box::new(CompositeReranker(
            configs.iter().map(build_reranker).collect(),
        )),
    }
}

async fn apply_composite_reranker(
    query: &[f32],
    results: Vec<SearchResult>,
    configs: &[RerankerConfig],
) -> VectorResult<Vec<SearchResult>> {
    if configs.is_empty() {
        return Ok(results);
    }

    let mut current = results;
    let mut aggregate_scores: HashMap<uuid::Uuid, f32> = HashMap::new();
    let mut by_id: HashMap<uuid::Uuid, SearchResult> = HashMap::new();

    for config in configs {
        let reranked = build_reranker(config).rerank(query, current).await?;
        let normalized = normalize_scores(reranked);
        let stage_weight = stage_weight(config);
        current = normalized.clone();

        for result in normalized {
            *aggregate_scores.entry(result.id).or_insert(0.0) += stage_weight * result.score;
            by_id.insert(result.id, result);
        }
    }

    let mut final_results = by_id
        .into_iter()
        .filter_map(|(id, mut result)| {
            let final_score = aggregate_scores.get(&id).copied()?;
            result.score = final_score;
            Some(result)
        })
        .collect::<Vec<_>>();
    sort_results_desc(&mut final_results);
    Ok(final_results)
}

fn normalize_scores(mut results: Vec<SearchResult>) -> Vec<SearchResult> {
    if results.is_empty() {
        return results;
    }

    let min_score = results
        .iter()
        .map(|result| result.score)
        .fold(f32::INFINITY, f32::min);
    let max_score = results
        .iter()
        .map(|result| result.score)
        .fold(f32::NEG_INFINITY, f32::max);
    let range = max_score - min_score;

    if range.abs() < f32::EPSILON {
        for result in &mut results {
            result.score = 1.0;
        }
        return results;
    }

    for result in &mut results {
        result.score = ((result.score - min_score) / range).clamp(0.0, 1.0);
    }
    results
}

fn stage_weight(config: &RerankerConfig) -> f32 {
    match config {
        RerankerConfig::None => 0.0,
        RerankerConfig::Diversity { weight, .. } => *weight,
        RerankerConfig::Recency { weight, .. } => *weight,
        RerankerConfig::Composite(_) => 1.0,
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

#[cfg(test)]
mod tests {
    use chrono::{Duration, Utc};

    use super::apply_reranker_config;
    use crate::types::{RerankerConfig, SearchResult};

    fn fixture_result(
        id: uuid::Uuid,
        vector: Vec<f32>,
        created_at: chrono::DateTime<Utc>,
    ) -> SearchResult {
        SearchResult {
            id,
            score: 0.5,
            vector: Some(vector),
            metadata: serde_json::json!({}),
            text: None,
            created_at,
        }
    }

    #[tokio::test]
    async fn composite_diversity_then_recency_is_deterministic() {
        let now = Utc::now();
        let first_id = uuid::Uuid::parse_str("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa").unwrap();
        let second_id = uuid::Uuid::parse_str("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb").unwrap();
        let third_id = uuid::Uuid::parse_str("cccccccc-cccc-cccc-cccc-cccccccccccc").unwrap();
        let query = vec![1.0, 0.0, 0.0];
        let results = vec![
            fixture_result(first_id, vec![0.95, 0.05, 0.0], now),
            fixture_result(second_id, vec![0.5, 0.5, 0.0], now - Duration::days(1)),
            fixture_result(third_id, vec![0.2, 0.8, 0.0], now - Duration::days(30)),
        ];

        let config = RerankerConfig::Composite(vec![
            RerankerConfig::Diversity {
                lambda: 0.8,
                weight: 0.6,
            },
            RerankerConfig::Recency {
                boost: 0.5,
                half_life_days: 7.0,
                weight: 0.4,
            },
        ]);

        let reranked = apply_reranker_config(&query, results, Some(&config))
            .await
            .unwrap();
        let ids = reranked.into_iter().map(|item| item.id).collect::<Vec<_>>();

        assert_eq!(ids, vec![first_id, second_id, third_id]);
    }
}
