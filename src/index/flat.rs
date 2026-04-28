// index/flat.rs — brute-force flat index for small collections (< 1 000 vectors).
use std::sync::RwLock;

use rayon::prelude::*;
use tracing::instrument;

use crate::{
    config::VectorConfig,
    error::{VectorError, VectorResult},
    index::hnsw::HnswIndex,
    types::DistanceMetric,
};

// ─── Distance kernels ────────────────────────────────────────────────────────

/// Cosine distance (1 − cosine similarity); SIMD-friendly iterator form.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 1.0 } else { 1.0 - dot / (na * nb) }
}

/// Euclidean (L2) distance.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt()
}

/// Negative dot product ("distance" — lower = more similar).
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
}

// ─── FlatIndex ───────────────────────────────────────────────────────────────

/// Brute-force flat index backed by a `RwLock<Vec<(id, vector)>>`.
pub struct FlatIndex {
    vectors: RwLock<Vec<(usize, Vec<f32>)>>,
    /// Expected vector dimensionality.
    pub dimensions: usize,
    /// Distance metric for similarity comparisons.
    pub distance: DistanceMetric,
}

impl FlatIndex {
    /// Create a new, empty flat index.
    pub fn new(dimensions: usize, distance: DistanceMetric) -> Self {
        FlatIndex { vectors: RwLock::new(Vec::new()), dimensions, distance }
    }

    /// Insert a single vector, validating its dimensionality.
    #[instrument(skip(self, vector))]
    pub fn insert(&self, id: usize, vector: Vec<f32>) -> VectorResult<()> {
        if vector.len() != self.dimensions {
            return Err(VectorError::DimensionMismatch { expected: self.dimensions, got: vector.len() });
        }
        self.vectors.write().map_err(|e| VectorError::Index(e.to_string()))?.push((id, vector));
        Ok(())
    }

    /// Insert multiple vectors.
    #[instrument(skip(self, items))]
    pub fn insert_batch(&self, items: Vec<(usize, Vec<f32>)>) -> VectorResult<()> {
        for (_, v) in &items {
            if v.len() != self.dimensions {
                return Err(VectorError::DimensionMismatch { expected: self.dimensions, got: v.len() });
            }
        }
        self.vectors.write().map_err(|e| VectorError::Index(e.to_string()))?.extend(items);
        Ok(())
    }

    /// Score all stored vectors in parallel and return the `top_k` closest.
    #[instrument(skip(self, query))]
    pub fn search(&self, query: &[f32], top_k: usize) -> VectorResult<Vec<(usize, f32)>> {
        if query.len() != self.dimensions {
            return Err(VectorError::DimensionMismatch { expected: self.dimensions, got: query.len() });
        }
        let vecs = self.vectors.read().map_err(|e| VectorError::Index(e.to_string()))?;
        let dist = self.distance;
        let mut scores: Vec<(usize, f32)> = vecs.par_iter().map(|(id, v)| {
            let d = match dist {
                DistanceMetric::Cosine => cosine_similarity(query, v),
                DistanceMetric::Euclidean => euclidean_distance(query, v),
                DistanceMetric::DotProduct => dot_product(query, v),
            };
            (*id, d)
        }).collect();
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        Ok(scores)
    }

    /// Remove a vector by id. Returns `true` if the id was present.
    #[instrument(skip(self))]
    pub fn delete(&self, id: usize) -> VectorResult<bool> {
        let mut vecs = self.vectors.write().map_err(|e| VectorError::Index(e.to_string()))?;
        let before = vecs.len();
        vecs.retain(|(vid, _)| *vid != id);
        Ok(vecs.len() < before)
    }

    /// Return the number of stored vectors.
    pub fn len(&self) -> usize {
        self.vectors.read().map(|v| v.len()).unwrap_or(0)
    }

    /// Return `true` if no vectors are stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return all stored (id, vector) pairs (used for persistence and migration).
    pub fn all_vectors(&self) -> VectorResult<Vec<(usize, Vec<f32>)>> {
        Ok(self.vectors.read().map_err(|e| VectorError::Index(e.to_string()))?.clone())
    }

    /// Migrate all vectors into a fresh [`HnswIndex`].
    #[instrument(skip(self, config))]
    pub fn to_hnsw(&self, config: &VectorConfig) -> VectorResult<HnswIndex> {
        let hnsw = HnswIndex::new_with_dimensions(config, self.distance, self.dimensions)?;
        let items = self.all_vectors()?;
        hnsw.insert_batch(&items)?;
        Ok(hnsw)
    }
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert_abs_diff_eq!(cosine_similarity(&a, &b), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn cosine_identical_vectors() {
        let a = vec![1.0f32, 1.0, 1.0];
        assert_abs_diff_eq!(cosine_similarity(&a, &a), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn euclidean_known_distance() {
        let a = vec![0.0f32, 0.0, 0.0];
        let b = vec![3.0f32, 4.0, 0.0];
        assert_abs_diff_eq!(euclidean_distance(&a, &b), 5.0, epsilon = 1e-6);
    }

    #[test]
    fn euclidean_same_point() {
        let a = vec![1.0f32, 2.0, 3.0];
        assert_abs_diff_eq!(euclidean_distance(&a, &a), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn dot_product_known() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        assert_abs_diff_eq!(dot_product(&a, &b), -32.0, epsilon = 1e-6);
    }

    #[test]
    fn flat_index_insert_search() {
        let idx = FlatIndex::new(2, DistanceMetric::Euclidean);
        idx.insert(0, vec![0.0, 0.0]).unwrap();
        idx.insert(1, vec![1.0, 1.0]).unwrap();
        idx.insert(2, vec![10.0, 10.0]).unwrap();
        let results = idx.search(&[0.1, 0.1], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn flat_index_delete() {
        let idx = FlatIndex::new(2, DistanceMetric::Euclidean);
        idx.insert(42, vec![1.0, 1.0]).unwrap();
        assert_eq!(idx.len(), 1);
        assert!(idx.delete(42).unwrap());
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn dimension_mismatch_returns_error() {
        let idx = FlatIndex::new(3, DistanceMetric::Euclidean);
        let err = idx.insert(0, vec![1.0, 2.0]).unwrap_err();
        assert!(err.is_dimension_mismatch());
    }
}
