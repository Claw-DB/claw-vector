// embeddings/types.rs — EmbeddingRequest, EmbeddingResponse, and EmbedVector domain types.
use serde::{Deserialize, Serialize};

/// A request to embed one or more texts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    /// Texts to embed.
    pub texts: Vec<String>,
    /// Override the model name (empty string = use service default).
    pub model_name: String,
    /// Whether to L2-normalise the output vectors.
    pub normalize: bool,
}

/// A single dense embedding vector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbedVector {
    /// Raw float values.
    pub values: Vec<f32>,
    /// Number of dimensions.
    pub dimensions: i32,
}

/// Response from the embedding service containing one vector per input text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// One embedding per input text, in the same order.
    pub vectors: Vec<EmbedVector>,
    /// The model that generated the embeddings.
    pub model_name: String,
    /// End-to-end latency of the embedding call in milliseconds.
    pub latency_ms: i64,
}

/// Metadata describing the currently loaded embedding model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Name of the model running in the embedding service.
    pub model_name: String,
    /// Number of output dimensions produced by the model.
    pub dimensions: usize,
    /// Maximum supported sequence length.
    pub max_sequence_length: usize,
    /// Runtime device, such as `cpu` or `cuda`.
    pub device: String,
}
