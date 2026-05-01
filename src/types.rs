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
            DistanceMetric::DotProduct => -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>(),
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
            metadata: serde_json::json!({}),
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
    /// Workspace identifier used for tenant isolation.
    pub workspace_id: String,
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
    /// Normalized similarity score in the range `[0.0, 1.0]` (higher is better).
    pub score: f32,
    /// The raw vector (only set when the query requests it).
    pub vector: Option<Vec<f32>>,
    /// Record metadata (only set when the query requests it).
    pub metadata: serde_json::Value,
    /// Original text (if stored with the record).
    pub text: Option<String>,
    /// UTC timestamp when the source record was created.
    pub created_at: DateTime<Utc>,
}

/// Additional metrics captured for a search operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SearchMetrics {
    /// Dimensionality of the input query vector.
    pub query_vector_dims: usize,
    /// Number of raw ANN or hybrid candidates examined before post-processing.
    pub candidates_evaluated: usize,
    /// Number of candidates that survived post-filtering and reranking.
    pub post_filter_count: usize,
    /// End-to-end latency for the search in microseconds.
    pub latency_us: u64,
}

/// Full search response, including user-visible results and execution metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SearchResponse {
    /// Ordered nearest-neighbour results.
    pub results: Vec<SearchResult>,
    /// Metrics captured while serving the request.
    pub metrics: SearchMetrics,
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
    /// Numeric greater-than check.
    Gt {
        /// Dot-notation JSON path to compare.
        key: String,
        /// Threshold value.
        value: f64,
    },
    /// Numeric less-than check.
    Lt {
        /// Dot-notation JSON path to compare.
        key: String,
        /// Threshold value.
        value: f64,
    },
    /// Case-insensitive substring match for string values.
    Contains {
        /// Dot-notation JSON path to compare.
        key: String,
        /// Substring to search for.
        value: String,
    },
    /// Membership check for scalar JSON values.
    In {
        /// Dot-notation JSON path to compare.
        key: String,
        /// Candidate values.
        values: Vec<serde_json::Value>,
    },
    /// Presence check for a key or nested path.
    Exists {
        /// Dot-notation JSON path whose presence is required.
        key: String,
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
        crate::search::filters::apply_filter(self, metadata)
    }
}

/// Post-retrieval reranking configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RerankerConfig {
    /// Disable reranking.
    None,
    /// Promote diversity using maximal marginal relevance.
    Diversity {
        /// Relevance-vs-diversity balance in the range `[0.0, 1.0]`.
        lambda: f32,
        /// Stage weight used by the composite reranker.
        weight: f32,
    },
    /// Boost recently created records.
    Recency {
        /// Strength of the recency boost.
        boost: f32,
        /// Exponential half-life in days.
        half_life_days: f32,
        /// Stage weight used by the composite reranker.
        weight: f32,
    },
    /// Apply multiple rerankers in sequence.
    Composite(Vec<RerankerConfig>),
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
    /// Optional post-retrieval reranking strategy.
    pub reranker: Option<RerankerConfig>,
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
        if let Some(filter) = &self.filter {
            crate::search::filters::validate_filter(filter)?;
        }
        Ok(())
    }
}

/// Hybrid vector + keyword search query.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HybridQuery {
    /// Target collection name.
    pub collection: String,
    /// Query vector used for ANN retrieval.
    pub vector: Vec<f32>,
    /// Optional keyword query used for FTS5 retrieval.
    pub text: Option<String>,
    /// Maximum number of results to return.
    pub top_k: usize,
    /// Blend factor where `1.0` is vector-only and `0.0` is keyword-only.
    pub alpha: f32,
    /// Optional metadata filter applied after fusion.
    pub filter: Option<MetadataFilter>,
    /// If `true`, each `SearchResult` will include the raw vector.
    pub include_vectors: bool,
    /// Optional post-retrieval reranking strategy.
    pub reranker: Option<RerankerConfig>,
}

impl HybridQuery {
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
        if !(0.0..=1.0).contains(&self.alpha) {
            return Err(VectorError::SearchError(
                "hybrid alpha must be between 0.0 and 1.0".into(),
            ));
        }
        if let Some(filter) = &self.filter {
            crate::search::filters::validate_filter(filter)?;
        }
        Ok(())
    }
}

/// Persisted storage statistics for a collection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CollectionStats {
    /// Number of vectors stored in the collection.
    pub vector_count: u64,
    /// Estimated on-disk size of the collection in bytes.
    pub size_bytes: u64,
}

/// Top-level runtime statistics for the vector engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct EngineStats {
    /// Number of known collections.
    pub collection_count: usize,
    /// Total vectors stored across all collections.
    pub total_vectors: u64,
    /// Number of indexes currently loaded in memory.
    pub loaded_indexes: usize,
    /// Number of mmap vector files currently opened.
    pub loaded_mmap_files: usize,
    /// Embedding cache hit counter.
    pub embedding_cache_hits: u64,
    /// Embedding cache miss counter.
    pub embedding_cache_misses: u64,
}
