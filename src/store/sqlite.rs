// store/sqlite.rs — SQLite persistence layer for VectorRecord and Collection metadata.
use sqlx::{SqlitePool, sqlite::SqliteConnectOptions};
use std::path::Path;
use uuid::Uuid;

use crate::{
    error::VectorResult,
    types::{Collection, VectorRecord},
};

/// Manages all SQLite read/write operations for collections and vector records.
pub struct SqliteStore {
    pool: SqlitePool,
}

impl SqliteStore {
    /// Open (or create) the SQLite database at `path`, applying schema migrations.
    pub async fn open(path: &Path) -> VectorResult<Self> {
        let opts = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true);
        let pool = SqlitePool::connect_with(opts).await?;
        let store = SqliteStore { pool };
        store.run_migrations().await?;
        Ok(store)
    }

    async fn run_migrations(&self) -> VectorResult<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS collections (
                name             TEXT PRIMARY KEY NOT NULL,
                dimensions       INTEGER NOT NULL,
                distance         TEXT NOT NULL,
                index_type       TEXT NOT NULL,
                created_at       TEXT NOT NULL,
                vector_count     INTEGER NOT NULL DEFAULT 0,
                metadata         TEXT NOT NULL DEFAULT 'null',
                ef_construction  INTEGER NOT NULL DEFAULT 200,
                m_connections    INTEGER NOT NULL DEFAULT 16
            );
            CREATE TABLE IF NOT EXISTS vector_records (
                id          TEXT PRIMARY KEY NOT NULL,
                collection  TEXT NOT NULL,
                vector      BLOB NOT NULL,
                metadata    TEXT NOT NULL DEFAULT 'null',
                text        TEXT,
                created_at  TEXT NOT NULL,
                FOREIGN KEY (collection) REFERENCES collections(name)
            );
            "#,
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Persist a new collection definition (upsert).
    pub async fn save_collection(&self, col: &Collection) -> VectorResult<()> {
        sqlx::query(
            r#"INSERT OR REPLACE INTO collections
               (name, dimensions, distance, index_type, created_at, vector_count, metadata,
                ef_construction, m_connections)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"#,
        )
        .bind(&col.name)
        .bind(col.dimensions as i64)
        .bind(serde_json::to_string(&col.distance)?)
        .bind(serde_json::to_string(&col.index_type)?)
        .bind(col.created_at.to_rfc3339())
        .bind(col.vector_count as i64)
        .bind(serde_json::to_string(&col.metadata)?)
        .bind(col.ef_construction as i64)
        .bind(col.m_connections as i64)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Delete a collection by name.
    pub async fn delete_collection(&self, name: &str) -> VectorResult<()> {
        sqlx::query("DELETE FROM collections WHERE name = ?")
            .bind(name)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// List all collection names.
    pub async fn list_collections(&self) -> VectorResult<Vec<serde_json::Value>> {
        let rows = sqlx::query_as::<_, (String,)>("SELECT name FROM collections ORDER BY name")
            .fetch_all(&self.pool)
            .await?;
        Ok(rows.into_iter().map(|(n,)| serde_json::json!({ "name": n })).collect())
    }

    /// Persist a vector record (upsert).
    pub async fn save_record(&self, record: &VectorRecord) -> VectorResult<()> {
        let vector_bytes = encode_f32_slice(&record.vector);
        sqlx::query(
            r#"INSERT OR REPLACE INTO vector_records
               (id, collection, vector, metadata, text, created_at)
               VALUES (?, ?, ?, ?, ?, ?)"#,
        )
        .bind(record.id.to_string())
        .bind(&record.collection)
        .bind(vector_bytes)
        .bind(serde_json::to_string(&record.metadata)?)
        .bind(&record.text)
        .bind(record.created_at.to_rfc3339())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Retrieve a single vector record by id.
    pub async fn get_record(&self, id: Uuid) -> VectorResult<Option<serde_json::Value>> {
        let row = sqlx::query_as::<_, (String, String, Vec<u8>, String, Option<String>, String)>(
            "SELECT id, collection, vector, metadata, text, created_at FROM vector_records WHERE id = ?",
        )
        .bind(id.to_string())
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|(id, collection, vec_bytes, metadata, text, created_at)| {
            serde_json::json!({
                "id": id,
                "collection": collection,
                "vector": decode_f32_slice(&vec_bytes),
                "metadata": metadata,
                "text": text,
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

fn encode_f32_slice(values: &[f32]) -> Vec<u8> {
    use byteorder::{LittleEndian, WriteBytesExt};
    let mut buf = Vec::with_capacity(values.len() * 4);
    for &v in values {
        buf.write_f32::<LittleEndian>(v).unwrap();
    }
    buf
}

fn decode_f32_slice(bytes: &[u8]) -> Vec<f32> {
    use byteorder::{LittleEndian, ReadBytesExt};
    let mut cursor = std::io::Cursor::new(bytes);
    let mut result = Vec::with_capacity(bytes.len() / 4);
    while let Ok(v) = cursor.read_f32::<LittleEndian>() {
        result.push(v);
    }
    result
}
