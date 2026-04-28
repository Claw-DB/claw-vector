// search/ann.rs — approximate nearest-neighbour search orchestration.
use crate::{
    error::VectorResult,
    index::selector::IndexSelector,
    types::{SearchQuery, SearchResult},
};

/// Orchestrates ANN search over an [`IndexSelector`].
pub struct AnnSearch;

impl AnnSearch {
    /// Execute a nearest-neighbour search and return filtered, ranked results.
    pub fn search(
        index: &IndexSelector,
        query: &SearchQuery,
        ef_search: usize,
        records: &std::collections::HashMap<usize, SearchResult>,
    ) -> VectorResult<Vec<SearchResult>> {
        query.validate()?;
        let raw = index.search(&query.vector, query.top_k, ef_search)?;

        let mut results: Vec<SearchResult> = raw
            .into_iter()
            .filter_map(|(id, score)| {
                let rec = records.get(&id)?;
                if let Some(filter) = &query.filter {
                    if !filter.matches(&rec.metadata) { return None; }
                }
                let mut result = rec.clone();
                result.score = score;
                if !query.include_vectors { result.vector = None; }
                if !query.include_metadata { result.metadata = serde_json::Value::Null; }
                Some(result)
            })
            .collect();

        results.truncate(query.top_k);
        Ok(results)
    }
}
