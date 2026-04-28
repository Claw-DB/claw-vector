// collections.rs — collection management operations (create/list/delete).
use crate::{
    engine::VectorEngine,
    error::VectorResult,
    types::{Collection, DistanceMetric},
};

/// Thin facade over [`VectorEngine`] for collection lifecycle management.
pub struct CollectionManager {
    engine: std::sync::Arc<VectorEngine>,
}

impl CollectionManager {
    /// Create a new manager backed by the given engine.
    pub fn new(engine: std::sync::Arc<VectorEngine>) -> Self {
        CollectionManager { engine }
    }

    /// Create a new collection.
    pub async fn create(
        &self,
        name: impl Into<String>,
        dimensions: usize,
        distance: DistanceMetric,
    ) -> VectorResult<Collection> {
        self.engine.create_collection(name, dimensions, distance).await
    }

    /// List all collections known to the store.
    pub async fn list(&self) -> VectorResult<Vec<serde_json::Value>> {
        self.engine.store().list_collections().await
    }
}
