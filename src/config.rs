// config.rs — VectorConfig with full builder pattern, defaults, validation, and env loading.
use std::{path::PathBuf, num::NonZeroUsize};
use serde::{Deserialize, Serialize};

use crate::error::{VectorError, VectorResult};

/// Runtime configuration for the claw-vector engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorConfig {
    /// Path to the SQLite database file for vector metadata.
    pub db_path: PathBuf,
    /// Directory for HNSW index files and mmap vector files.
    pub index_dir: PathBuf,
    /// gRPC endpoint of the Python embedding service (e.g. `"http://localhost:50051"`).
    pub embedding_service_url: String,
    /// Default embedding dimensionality (384 = all-MiniLM-L6-v2).
    pub default_dimensions: usize,
    /// HNSW `ef_construction` build parameter (higher → better recall, slower build).
    pub ef_construction: usize,
    /// HNSW `M` connections parameter (higher → better recall, more memory).
    pub m_connections: usize,
    /// HNSW `ef` search parameter (higher → better recall, slower search).
    pub ef_search: usize,
    /// Maximum number of vectors per index.
    pub max_elements: usize,
    /// Number of embedding LRU cache entries.
    pub cache_size: usize,
    /// Maximum number of texts per embedding gRPC call.
    pub batch_size: usize,
    /// Timeout for embedding gRPC calls in milliseconds.
    pub embedding_timeout_ms: u64,
    /// Number of rayon threads for parallel index operations.
    pub num_threads: usize,
}

impl Default for VectorConfig {
    fn default() -> Self {
        VectorConfig {
            db_path: PathBuf::from("claw_vector.db"),
            index_dir: PathBuf::from("claw_vector_indices"),
            embedding_service_url: "http://localhost:50051".into(),
            default_dimensions: 384,
            ef_construction: 200,
            m_connections: 16,
            ef_search: 50,
            max_elements: 1_000_000,
            cache_size: 10_000,
            batch_size: 64,
            embedding_timeout_ms: 5_000,
            num_threads: std::thread::available_parallelism()
                .unwrap_or(NonZeroUsize::new(4).unwrap())
                .get(),
        }
    }
}

impl VectorConfig {
    /// Return a new builder initialised with the default configuration.
    pub fn builder() -> VectorConfigBuilder {
        VectorConfigBuilder::default()
    }

    /// Load configuration from environment variables, falling back to defaults.
    ///
    /// Recognised variables:
    /// - `CLAW_VECTOR_DB_PATH`
    /// - `CLAW_VECTOR_INDEX_DIR`
    /// - `CLAW_EMBEDDING_URL`
    pub fn from_env() -> Self {
        let mut cfg = VectorConfig::default();
        if let Ok(v) = std::env::var("CLAW_VECTOR_DB_PATH") {
            cfg.db_path = PathBuf::from(v);
        }
        if let Ok(v) = std::env::var("CLAW_VECTOR_INDEX_DIR") {
            cfg.index_dir = PathBuf::from(v);
        }
        if let Ok(v) = std::env::var("CLAW_EMBEDDING_URL") {
            cfg.embedding_service_url = v;
        }
        cfg
    }
}

// ─── Builder ─────────────────────────────────────────────────────────────────

/// Fluent builder for [`VectorConfig`].
#[derive(Debug, Clone)]
pub struct VectorConfigBuilder {
    inner: VectorConfig,
}

impl Default for VectorConfigBuilder {
    fn default() -> Self {
        VectorConfigBuilder {
            inner: VectorConfig::default(),
        }
    }
}

impl VectorConfigBuilder {
    /// Set the SQLite database path.
    pub fn db_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.inner.db_path = path.into();
        self
    }

    /// Set the index directory.
    pub fn index_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.inner.index_dir = dir.into();
        self
    }

    /// Set the embedding service gRPC URL.
    pub fn embedding_service_url(mut self, url: impl Into<String>) -> Self {
        self.inner.embedding_service_url = url.into();
        self
    }

    /// Set the default embedding dimensionality.
    pub fn default_dimensions(mut self, dims: usize) -> Self {
        self.inner.default_dimensions = dims;
        self
    }

    /// Set the HNSW `ef_construction` parameter.
    pub fn ef_construction(mut self, ef: usize) -> Self {
        self.inner.ef_construction = ef;
        self
    }

    /// Set the HNSW `M` connections parameter.
    pub fn m_connections(mut self, m: usize) -> Self {
        self.inner.m_connections = m;
        self
    }

    /// Set the HNSW `ef_search` parameter.
    pub fn ef_search(mut self, ef: usize) -> Self {
        self.inner.ef_search = ef;
        self
    }

    /// Set the maximum number of vectors per index.
    pub fn max_elements(mut self, n: usize) -> Self {
        self.inner.max_elements = n;
        self
    }

    /// Set the LRU embedding cache capacity.
    pub fn cache_size(mut self, n: usize) -> Self {
        self.inner.cache_size = n;
        self
    }

    /// Set the maximum batch size for embedding calls.
    pub fn batch_size(mut self, n: usize) -> Self {
        self.inner.batch_size = n;
        self
    }

    /// Set the embedding gRPC call timeout in milliseconds.
    pub fn embedding_timeout_ms(mut self, ms: u64) -> Self {
        self.inner.embedding_timeout_ms = ms;
        self
    }

    /// Set the number of rayon threads.
    pub fn num_threads(mut self, n: usize) -> Self {
        self.inner.num_threads = n;
        self
    }

    /// Validate and return the completed [`VectorConfig`].
    ///
    /// # Errors
    /// - `dimensions` must be ≥ 1
    /// - `ef_construction` must be ≥ `m_connections`
    /// - `m_connections` must be ≥ 2
    pub fn build(self) -> VectorResult<VectorConfig> {
        let cfg = self.inner;
        if cfg.default_dimensions < 1 {
            return Err(VectorError::Config(
                "default_dimensions must be >= 1".into(),
            ));
        }
        if cfg.m_connections < 2 {
            return Err(VectorError::Config("m_connections must be >= 2".into()));
        }
        if cfg.ef_construction < cfg.m_connections {
            return Err(VectorError::Config(
                "ef_construction must be >= m_connections".into(),
            ));
        }
        Ok(cfg)
    }
}
