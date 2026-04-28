// store/sqlite.rs — SQLite persistence layer for VectorRecord and Collection metadata.
use sqlx::{SqlitePool, sqlite::SqliteConnectOptions};
use std::path::Path;
use uuid::Uuid;

use crate::{
    error::{VectorError, VectorResult},
    types::{Collection, VectorRecord},
};

/// Manages all SQLite read/write operations for collections and vector records.
pub struct VectorStore {
    pool: SqlitePool,
}

/// Backward-compatible alias for [`VectorStore`].
pub type SqliteStore = VectorStore;

impl VectorStore {
    /// Open (or create) the SQLite database at `path`, applying schema migrations.
    pub async fn new(path: &Path) -> VectorResult<Self> {
        let opts = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true);
        let pool = SqlitePool::connect_with(opts).await?;
        let store = VectorStore { pool };
        sqlx::migrate!().run(&store.pool).await.map_err(|e| VectorError::Store(e.into()))?;
        Ok(store)
    }

    /// Alias for [`VectorStore::new`].
    pub async fn open(path: &Path) -> VectorResult<Self> {
        Self::new(path).await
    }

    /// Persist a new collection definition (upsert).
    pub async fn create_collection(&self, col: &Collection) -> VectorResult<()> {
        sqlx::query(
            r#"INSERT OR REPLACE INTO collections
               (name, dimensions, distance, index_type, ef_construction, m_connections,
                created_at, vector_count, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"#,
        )
        .bind(&col.name)
        .bind(col.dimensions as i64)
        .bind(serde_json::to_string(&col.distance)?)
        .bind(serde_json::to_string(&col.index_type)?)
        .bind(col.ef_construction as i64)
        .bind(col.m_connections as i64)
        .bind(col.created_at.to_rfc3339())
        .bind(col.vector_count as i64)
        .bind(serde_json::to_string(&col.metadata)?)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Alias for [`VectorStore::create_collection`].
    pub async fn save_collection(&self, col: &Collection) -> VectorResult<()> {
        self.create_collection(col).await
    }

    /// Retrieve a collection by name.
    pub async fn get_collection(&self, name: &str) -> VectorResult<Collection> {
        type Row = (String, i64, String, String, i64, i64, String, i64, String);
        let row = sqlx::query_as::<_, Row>(
            "SELECT name, dimensions, distance, index_type, ef_construction, m_connections, \
             created_at, vector_count, metadata FROM collections WHERE name = ?",
        )
        .bind(name)
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some(row) => collection_from_row(row),
            None => Err(VectorError::NotFound {
                entity: "collection".into(),
                id: name.to_string(),
            }),
        }
    }

    /// Delete a collection by name.
    pub async fn delete_collection(&self, name: &str) -> VectorResult<()> {
        sqlx::query("DELETE FROM collections WHERE name = ?")
            .bind(name)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// List all collections.
    pub async fn list_collections(&self) -> VectorResult<Vec<Collection>> {
        type Row = (String, i64, String, String, i64, i64, String, i64, String);
        let rows = sqlx::query_as::<_, Row>(
            "SELECT name, dimensions, distance, index_type, ef_construction, m_connections, \
             created_at, vector_count, metadata FROM collections ORDER BY name",
        )
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter().map(collection_from_row).collect()
    }

    /// Persist a vector record (upsert), linking it to the given `internal_id`.
    pub async fn save_record(&self, record: &VectorRecord, internal_id: usize) -> VectorResult<()> {
        sqlx::query(
            r#"INSERT OR REPLACE INTO vector_records
               (id, internal_id, collection, text, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?)"#,
        )
        .bind(record.id.to_string())
        .bind(internal_id as i64)
        .bind(&record.collection)
        .bind(&record.text)
        .bind(serde_json::to_string(&record.metadata)?)
        .bind(record.created_at.to_rfc3339())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Retrieve a single vector record's metadata by id.
    pub async fn get_record(&self, id: Uuid) -> VectorResult<Option<serde_json::Value>> {
        let row = sqlx::query_as::<_, (String, i64, String, Option<String>, String, String)>(
            "SELECT id, internal_id, collection, text, metadata, created_at \
             FROM vector_records WHERE id = ?",
        )
        .bind(id.to_string())
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|(id, internal_id, collection, text, metadata, created_at)| {
            serde_json::json!({
                "id": id,
                "internal_id": internal_id,
                "collection": collection,
                "text": text,
                "metadata": metadata,
                "created_at": created_at,
            })
        }))
    }

    /// Delete a vector record by id. Returns `true` if it existed.
    pub async fn delete_record(&self, id: Uuid) -> VectorResult<bool> {
        let result = sqlx::query("DELETE FROM vector_records WHERE id = ?")
            .bind(id.to_string())
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected() > 0)
    }
}

/// Convert a raw database row into a [`Collection`], parsing JSON and RFC-3339 fields.
fn collection_from_row(
    (name, dimensions, distance, index_type, ef_construction, m_connections,
     created_at, vector_count, metadata): (String, i64, String, String, i64, i64, String, i64, String),
) -> VectorResult<Collection> {
    Ok(Collection {
        name,
        dimensions: dimensions as usize,
        distance: serde_json::from_str(&distance)?,
        index_type: serde_json::from_str(&index_type)?,
        ef_construction: ef_construction as usize,
        m_connections: m_connections as usize,
        created_at: chrono::DateTime::parse_from_rfc3339(&created_at)
            .map_err(|e| VectorError::Index(format!("invalid timestamp in DB: {e}")))?
            .with_timezone(&chrono::Utc),
        vector_count: vector_count as u64,
        metadata: serde_json::from_str(&metadata)?,
    })
}
