// error.rs — VectorError enum (thiserror) + VectorResult<T> + From<VectorError> for tonic::Status.
use thiserror::Error;

/// Unified error type for all claw-vector operations.
#[derive(Debug, Error)]
pub enum VectorError {
    /// HNSW or flat index failure.
    #[error("index error: {0}")]
    Index(String),

    /// SQLite persistence failure.
    #[error("store error: {0}")]
    Store(#[from] sqlx::Error),

    /// Embedding generation failure.
    #[error("embedding error: {0}")]
    Embedding(String),

    /// gRPC transport failure.
    #[error("gRPC error: {0}")]
    Grpc(#[from] tonic::Status),

    /// A collection-level error (not found, already exists, etc.).
    #[error("collection '{name}': {reason}")]
    Collection {
        /// Name of the collection involved.
        name: String,
        /// Human-readable explanation.
        reason: String,
    },

    /// Vector dimension does not match the collection's configured dimension.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Dimension required by the collection.
        expected: usize,
        /// Dimension of the supplied vector.
        got: usize,
    },

    /// JSON serialisation/deserialisation failure.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// I/O error (file system operations).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid configuration.
    #[error("config error: {0}")]
    Config(String),

    /// ANN search failure.
    #[error("search error: {0}")]
    SearchError(String),

    /// Metadata filter DSL error.
    #[error("filter error: {0}")]
    FilterError(String),

    /// The requested entity was not found.
    #[error("{entity} with id '{id}' not found")]
    NotFound {
        /// Type of entity (e.g. "record", "collection").
        entity: String,
        /// Identifier that was looked up.
        id: String,
    },
}

impl VectorError {
    /// Return `true` if this error represents a "not found" condition.
    pub fn is_not_found(&self) -> bool {
        matches!(self, VectorError::NotFound { .. })
    }

    /// Return `true` if this error is a dimension mismatch.
    pub fn is_dimension_mismatch(&self) -> bool {
        matches!(self, VectorError::DimensionMismatch { .. })
    }

    /// Return `true` if this error is a collection-level error.
    pub fn is_collection_error(&self) -> bool {
        matches!(self, VectorError::Collection { .. })
    }
}

impl From<VectorError> for tonic::Status {
    fn from(e: VectorError) -> tonic::Status {
        match &e {
            VectorError::NotFound { entity, id } => {
                tonic::Status::not_found(format!("{entity} '{id}' not found"))
            }
            VectorError::DimensionMismatch { expected, got } => tonic::Status::invalid_argument(
                format!("dimension mismatch: expected {expected}, got {got}"),
            ),
            VectorError::Collection { name, reason } => {
                tonic::Status::invalid_argument(format!("collection '{name}': {reason}"))
            }
            VectorError::Config(msg) => tonic::Status::invalid_argument(msg.clone()),
            VectorError::Embedding(msg) => tonic::Status::internal(msg.clone()),
            VectorError::Index(msg) => tonic::Status::internal(msg.clone()),
            VectorError::SearchError(msg) => tonic::Status::internal(msg.clone()),
            VectorError::FilterError(msg) => tonic::Status::internal(msg.clone()),
            VectorError::Grpc(status) => status.clone(),
            _ => tonic::Status::internal(e.to_string()),
        }
    }
}

/// Convenience result alias used throughout claw-vector.
pub type VectorResult<T> = Result<T, VectorError>;
