// search/hybrid.rs — hybrid ANN + keyword search fusion.
use std::{cmp::Ordering, collections::HashMap, sync::Arc, time::Instant};

use tracing::instrument;
use uuid::Uuid;

use crate::{
    error::VectorResult,
    search::{ann::AnnSearcher, filters::apply_filter},
    store::VectorStore,
    types::{HybridQuery, SearchMetrics, SearchQuery, SearchResponse, SearchResult},
};

/// Hybrid retrieval service combining ANN and SQLite FTS5 keyword search.
pub struct HybridSearcher {
    /// ANN search service.
    pub ann: Arc<AnnSearcher>,
    /// Persistent metadata and FTS store.
    pub store: Arc<VectorStore>,
}

impl HybridSearcher {
    /// Create a new hybrid searcher.
    pub fn new(ann: Arc<AnnSearcher>, store: Arc<VectorStore>) -> Self {
        Self { ann, store }
    }

    /// Execute hybrid search and return fused ranked results.
    #[instrument(skip(self, query))]
    pub async fn search(&self, query: HybridQuery) -> VectorResult<SearchResponse> {
        let workspace_id = self.ann.collection_manager.config.default_workspace_id.clone();
        self.search_in_workspace(&workspace_id, query).await
    }

    /// Execute hybrid search scoped to a workspace.
    #[instrument(skip(self, query))]
    pub async fn search_in_workspace(
        &self,
        workspace_id: &str,
        query: HybridQuery,
    ) -> VectorResult<SearchResponse> {
        query.validate()?;

        let started = Instant::now();
        let ann_candidates = if query.alpha <= 0.0 {
            0
        } else {
            ((query.top_k as f32) * (1.0 + query.alpha)).ceil() as usize
        }
        .max(query.top_k);
        let keyword_candidates = if query.text.as_deref().unwrap_or_default().trim().is_empty() {
            0
        } else {
            ((query.top_k as f32) * (2.0 - query.alpha)).ceil() as usize
        }
        .max(query.top_k);

        let ann_query = SearchQuery {
            collection: query.collection.clone(),
            vector: query.vector.clone(),
            top_k: ann_candidates,
            filter: None,
            include_vectors: query.include_vectors,
            include_metadata: true,
            ef_search: None,
            reranker: None,
        };

        let text = query.text.clone();
        let keyword_future = async {
            match text.as_deref() {
                Some(text) if !text.trim().is_empty() => {
                    self.store
                        .keyword_search(workspace_id, &query.collection, text, keyword_candidates)
                        .await
                }
                _ => Ok(Vec::new()),
            }
        };

        let ann_future = async {
            if query.alpha <= 0.0 {
                Ok(SearchResponse::default())
            } else {
                self.ann.search_in_workspace(workspace_id, ann_query).await
            }
        };

        let (ann_response, keyword_rows) = tokio::join!(ann_future, keyword_future);
        let ann_response = ann_response?;
        let keyword_rows = keyword_rows?;

        let keyword_results = self
            .build_keyword_results(
                workspace_id,
                &query.collection,
                &keyword_rows,
                query.include_vectors,
            )
            .await?;

        let mut fused = fuse_results(query.alpha, ann_response.results, keyword_results);
        fused.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
        });

        if let Some(filter) = &query.filter {
            fused.retain(|result| apply_filter(filter, &result.metadata));
        }

        let post_filter_count = fused.len();
        fused.truncate(query.top_k);

        Ok(SearchResponse {
            metrics: SearchMetrics {
                query_vector_dims: query.vector.len(),
                candidates_evaluated: ann_response.metrics.candidates_evaluated
                    + keyword_rows.len(),
                post_filter_count,
                latency_us: started.elapsed().as_micros() as u64,
            },
            results: fused,
        })
    }

    async fn build_keyword_results(
        &self,
        workspace_id: &str,
        collection: &str,
        rows: &[(usize, crate::types::VectorRecord, f32)],
        include_vectors: bool,
    ) -> VectorResult<Vec<SearchResult>> {
        let total = rows.len().max(1) as f32;
        let mut results = Vec::with_capacity(rows.len());

        for (rank_index, (internal_id, record, raw_rank)) in rows.iter().enumerate() {
            let rank_score = 1.0 - (rank_index as f32 / total);
            let bm25_score = (1.0 / (1.0 + raw_rank.abs())).clamp(0.0, 1.0);
            let vector = if include_vectors {
                Some(
                    self.ann
                        .collection_manager
                        .read_vector_by_internal_id(workspace_id, collection, *internal_id)
                        .await?,
                )
            } else {
                None
            };

            results.push(SearchResult {
                id: record.id,
                score: ((rank_score + bm25_score) / 2.0).clamp(0.0, 1.0),
                vector,
                metadata: record.metadata.clone(),
                text: record.text.clone(),
                created_at: record.created_at,
            });
        }

        Ok(results)
    }
}

/// Reciprocal rank fusion score helper exposed for unit tests.
pub(crate) fn rrf_score(rank: usize, k: f32) -> f32 {
    1.0 / (k + rank.max(1) as f32)
}

fn fuse_results(
    alpha: f32,
    ann: Vec<SearchResult>,
    keyword: Vec<SearchResult>,
) -> Vec<SearchResult> {
    #[derive(Default)]
    struct Entry {
        result: Option<SearchResult>,
        ann_score: Option<f32>,
        keyword_score: Option<f32>,
        ann_rank: Option<usize>,
        keyword_rank: Option<usize>,
    }

    let mut entries = HashMap::<Uuid, Entry>::new();
    for (rank, result) in ann.into_iter().enumerate() {
        let entry = entries.entry(result.id).or_default();
        entry.result = Some(result.clone());
        entry.ann_score = Some(result.score);
        entry.ann_rank = Some(rank + 1);
    }

    for (rank, result) in keyword.into_iter().enumerate() {
        let entry = entries.entry(result.id).or_default();
        if entry.result.is_none() {
            entry.result = Some(result.clone());
        }
        entry.keyword_score = Some(result.score);
        entry.keyword_rank = Some(rank + 1);
    }

    entries
        .into_values()
        .filter_map(|mut entry| {
            let mut result = entry.result.take()?;
            let ann_rrf = entry
                .ann_rank
                .map(|rank| rrf_score(rank, 60.0))
                .unwrap_or(0.0);
            let keyword_rrf = entry
                .keyword_rank
                .map(|rank| rrf_score(rank, 60.0))
                .unwrap_or(0.0);
            result.score = match (entry.ann_score, entry.keyword_score) {
                (Some(ann_score), Some(keyword_score)) => {
                    alpha * ann_score + (1.0 - alpha) * keyword_score
                }
                (Some(ann_score), None) => alpha * ann_score + (1.0 - alpha) * ann_rrf,
                (None, Some(keyword_score)) => alpha * keyword_rrf + (1.0 - alpha) * keyword_score,
                (None, None) => 0.0,
            };
            result.score += ann_rrf + keyword_rrf;
            Some(result)
        })
        .collect()
}
