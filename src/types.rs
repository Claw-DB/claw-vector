// types.rs — core domain types: VectorRecord, Collection, DistanceMetric, IndexType,
//             SearchResult, SearchQuery, and MetadataFilter.
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{VectorError, VectorResult};

// ─── DistanceMetric ───────────────────────────────────────────────────────────

/// Distance metric used to compare vectors in a collection.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistanceMetric {
    /// Cosine similarity (1 − cosine → lower is closer).
    Cosine,
    /// Euclidean (L2) distance.
    Euclidean,
    /// Negative dot product (lower is closer).
    DotProduct,
}

impl DistanceMetric {
    /// Compute the distance between two vectors under this metric.
    pub fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            DistanceMetric::Cosine => {
                let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                if na == 0.0 || nb == 0.0 {
                    1.0
                } else {
                    1.0 - dot / (na * nb)
                }
            }
            DistanceMetric::Euclidean => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f32>()
                .sqrt(),
            DistanceMetric::DotProduct => {
                -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
            }
        }
    }
}

// ─── IndexType ───────────────────────────────────────────────────────────────

/// The backing index algorithm for a collection.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexType {
    /// Approximate nearest-neighbour search via HNSW.
    HNSW,
    /// Brute-force flat scan (best for small collections).
    Flat,
}

impl IndexType {
    /// Choose the appropriate index type based on the collection size.
    ///
    /// Returns `Flat` when `vector_count < 1 000`, `HNSW` otherwise.
    pub fn auto_select(vector_count: usize) -> Self {
        if vector_count < 1_000 {
            IndexType::Flat
        } else {
            IndexType::HNSW
        }
    }
}

// ─── VectorRecord ────────────────────────────────────────────────────────────

/// A single vector stored in a collection, with optional text and metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorRecord {
    /// Globally unique record identifier.
    pub id: Uuid,
    /// Name of the collection this record belongs to.
    pub collection: String,
    /// The raw embedding vector.
    pub vector: Vec<f32>,
    /// Arbitrary JSON metadata attached to this record.
    pub metadata: serde_json::Value,
    /// Original text from which the vector was generated (if stored).
    pub text: Option<String>,
    /// UTC timestamp of record creation.
    pub created_at: DateTime<Utc>,
}

impl VectorRecord {
    /// Create a new record with a fresh UUID and no text or metadata.
    pub fn new(collection: impl Into<String>, vector: Vec<f32>) -> Self {
        VectorRecord {
            id: Uuid::new_v4(),
            collection: collection.into(),
            vector,
            metadata: serde_json::Value::Null,
            text: None,
            created_at: Utc::now(),
        }
    }

    /// Builder: attach the original text to this record.
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    /// Builder: attach arbitrary JSON metadata to this record.
    pub fn with_metadata(mut self, meta: serde_json::Value) -> Self {
        self.metadata = meta;
        self
    }

    /// Return the dimensionality of the stored vector.
    pub fn dimensions(&self) -> usize {
        self.vector.len()
    }
}

// ─── Collection ──────────────────────────────────────────────────────────────

/// Describes a named collection of vectors with a shared dimension and distance metric.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Collection {
    /// Unique collection name.
    pub name: String,
    /// Expected vector dimensionality for all records in this collection.
    pub dimensions: usize,
    /// Distance metric used for similarity search.
    pub distance: DistanceMetric,
    /// Active index algorithm.
    pub index_type: IndexType,
    /// UTC timestamp of collection creation.
    pub created_at: DateTime<Utc>,
    /// Number of vectors currently stored in this collection.
    pub vector_count: u64,
    /// Arbitrary JSON metadata for the collection.
    pub metadata: serde_json::Value,
    /// HNSW `ef_construction` build parameter.
    pub ef_construction: usize,
    /// HNSW `M` connections parameter.
    pub m_connections: usize,
}

// ─── SearchResult ────────────────────────────────────────────────────────────

/// A single result returned by a nearest-neighbour search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResult {
    /// Record identifier.
    pub id: Uuid,
    /// Distance or similarity score (lower = closer for distance metrics).
    pub score: f32,
    /// The raw vector (only set when the query requests it).
    pub vector: Option<Vec<f32>>,
    /// Record metadata (only set when the query requests it).
    pub metadata: serde_json::Value,
    /// Original text (if stored with the record).
    pub text: Option<String>,
}

// ─── MetadataFilter ──────────────────────────────────────────────────────────

/// A composable DSL for filtering search results by their JSON metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum MetadataFilter {
    /// Equality check: `metadata[key] == value`.
    Eq {
        /// JSON object key to compare.
        key: String,
        /// Expected value.
        value: serde_json::Value,
    },
    /// Logical AND of multiple sub-filters.
    And(Vec<MetadataFilter>),
    /// Logical OR of multiple sub-filters.
    Or(Vec<MetadataFilter>),
    /// Logical NOT of a sub-filter.
    Not(Box<MetadataFilter>),
}

impl MetadataFilter {
    /// Evaluate this filter against a JSON metadata object.
    pub fn matches(&self, metadata: &serde_json::Value) -> bool {
        match self {
            MetadataFilter::Eq { key, value } => metadata.get(key) == Some(value),
            MetadataFilter::And(filters) => filters.iter().all(|f| f.matches(metadata)),
            MetadataFilter::Or(filters) => filters.iter().any(|f| f.matches(metadata)),
            MetadataFilter::Not(filter) => !filter.matches(metadata),
        }
    }
}

// ─── SearchQuery ─────────────────────────────────────────────────────────────

/// A nearest-neighbour search query.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchQuery {
    /// Target collection name.
    pub collection: String,
    /// Query vector.
    pub vector: Vec<f32>,
    /// Maximum number of results to return.
    pub top_k: usize,
    /// Optional metadata filter applied after ANN retrieval.
    pub filter: Option<MetadataFilter>,
    /// If `true`, each `SearchResult` will include the raw vector.
    pub include_vectors: bool,
    /// If `true`, each `SearchResult` will include the JSON metadata.
    pub include_metadata: bool,
    /// Override the HNSW `ef_search` parameter for this query.
    pub ef_search: Option<usize>,
}

impl SearchQuery {
    /// Validate the query fields, returning an error for invalid configurations.
    pub fn validate(&self) -> VectorResult<()> {
        if self.collection.is_empty() {
            return Err(VectorError::SearchError(
                "collection name must not be empty".into(),
            ));
        }
        if self.vector.is_empty() {
            return Err(VectorError::SearchError(
                "query vector must not be empty".into(),
            ));
        }
        if self.top_k == 0 {
            return Err(VectorError::SearchError("top_k must be > 0".into()));
        }
        Ok(())
    }
}
