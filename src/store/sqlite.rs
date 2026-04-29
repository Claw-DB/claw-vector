// store/sqlite.rs — SQLite persistence layer for VectorRecord and Collection metadata.
use std::{collections::HashMap, path::Path};

use sqlx::{
    sqlite::{SqliteConnectOptions, SqlitePoolOptions},
    QueryBuilder, Sqlite, SqlitePool,
};
use uuid::Uuid;

use crate::{
    error::{VectorError, VectorResult},
    types::{Collection, CollectionStats, DistanceMetric, IndexType, VectorRecord},
};

/// Manages all SQLite read/write operations for collections and vector records.
pub struct VectorStore {
    pool: SqlitePool,
}

/// Backward-compatible alias for [`VectorStore`].
pub type SqliteStore = VectorStore;

impl VectorStore {
    /// Open (or create) the SQLite database at `db_path`, applying schema migrations.
    pub async fn new(db_path: &Path) -> VectorResult<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let options = SqliteConnectOptions::new()
            .filename(db_path)
            .create_if_missing(true)
            .foreign_keys(true);

        let pool = SqlitePoolOptions::new()
            .max_connections(8)
            .connect_with(options)
            .await?;

        sqlx::query("PRAGMA journal_mode = WAL")
            .execute(&pool)
            .await?;
        sqlx::query("PRAGMA synchronous = NORMAL")
            .execute(&pool)
            .await?;
        sqlx::query("PRAGMA temp_store = MEMORY")
            .execute(&pool)
            .await?;

        sqlx::migrate!()
            .run(&pool)
            .await
            .map_err(|err| VectorError::Index(format!("failed to run SQLite migrations: {err}")))?;

        Ok(VectorStore { pool })
    }

    /// Alias for [`VectorStore::new`].
    pub async fn open(path: &Path) -> VectorResult<Self> {
        Self::new(path).await
    }

    /// Return the underlying SQLx connection pool.
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Persist a new collection definition (upsert).
    pub async fn create_collection(&self, col: &Collection) -> VectorResult<()> {
        sqlx::query(
            r#"INSERT INTO collections
               (name, dimensions, distance, index_type, ef_construction, m_connections,
                created_at, vector_count, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"#,
        )
        .bind(&col.name)
        .bind(col.dimensions as i64)
        .bind(distance_to_db(col.distance))
        .bind(index_type_to_db(col.index_type))
        .bind(col.ef_construction as i64)
        .bind(col.m_connections as i64)
        .bind(col.created_at.to_rfc3339())
        .bind(col.vector_count as i64)
        .bind(normalize_metadata(&col.metadata)?)
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
        let row = sqlx::query_as::<_, CollectionRow>(
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
        let mut tx = self.pool.begin().await?;
        sqlx::query("DELETE FROM vector_records WHERE collection = ?")
            .bind(name)
            .execute(&mut *tx)
            .await?;
        sqlx::query("DELETE FROM collections WHERE name = ?")
            .bind(name)
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        Ok(())
    }

    /// List all collections.
    pub async fn list_collections(&self) -> VectorResult<Vec<Collection>> {
        let rows = sqlx::query_as::<_, CollectionRow>(
            "SELECT name, dimensions, distance, index_type, ef_construction, m_connections, \
             created_at, vector_count, metadata FROM collections ORDER BY name",
        )
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter().map(collection_from_row).collect()
    }

    /// Persist a vector record, linking it to the given `internal_id`.
    pub async fn insert_record(
        &self,
        record: &VectorRecord,
        internal_id: usize,
    ) -> VectorResult<()> {
        sqlx::query(
            r#"INSERT INTO vector_records
               (id, internal_id, collection, text, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?)"#,
        )
        .bind(record.id.to_string())
        .bind(internal_id as i64)
        .bind(&record.collection)
        .bind(&record.text)
        .bind(normalize_metadata(&record.metadata)?)
        .bind(record.created_at.to_rfc3339())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Alias for [`VectorStore::insert_record`].
    pub async fn save_record(&self, record: &VectorRecord, internal_id: usize) -> VectorResult<()> {
        self.insert_record(record, internal_id).await
    }

    /// Retrieve a record and its internal identifier by UUID.
    pub async fn get_record(&self, id: Uuid) -> VectorResult<(VectorRecord, usize)> {
        let row = sqlx::query_as::<_, RecordRow>(
            "SELECT id, internal_id, collection, text, metadata, created_at \
             FROM vector_records WHERE id = ?",
        )
        .bind(id.to_string())
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some(row) => record_from_row(row),
            None => Err(VectorError::NotFound {
                entity: "record".into(),
                id: id.to_string(),
            }),
        }
    }

    /// Delete a vector record by id and return its previous internal id when found.
    pub async fn delete_record(&self, id: Uuid) -> VectorResult<Option<usize>> {
        let mut tx = self.pool.begin().await?;
        let internal_id =
            sqlx::query_scalar::<_, i64>("SELECT internal_id FROM vector_records WHERE id = ?")
                .bind(id.to_string())
                .fetch_optional(&mut *tx)
                .await?
                .map(|value| value as usize);

        if internal_id.is_some() {
            sqlx::query("DELETE FROM vector_records WHERE id = ?")
                .bind(id.to_string())
                .execute(&mut *tx)
                .await?;
        }

        tx.commit().await?;
        Ok(internal_id)
    }

    /// Insert multiple vector records in a single transaction.
    pub async fn batch_insert_records(
        &self,
        records: &[(VectorRecord, usize)],
    ) -> VectorResult<()> {
        let mut tx = self.pool.begin().await?;
        for (record, internal_id) in records {
            sqlx::query(
                r#"INSERT INTO vector_records
                   (id, internal_id, collection, text, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)"#,
            )
            .bind(record.id.to_string())
            .bind(*internal_id as i64)
            .bind(&record.collection)
            .bind(&record.text)
            .bind(normalize_metadata(&record.metadata)?)
            .bind(record.created_at.to_rfc3339())
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    /// Resolve a record UUID to its internal id.
    pub async fn uuid_to_internal(&self, id: Uuid) -> VectorResult<usize> {
        let internal_id =
            sqlx::query_scalar::<_, i64>("SELECT internal_id FROM vector_records WHERE id = ?")
                .bind(id.to_string())
                .fetch_optional(&self.pool)
                .await?
                .ok_or_else(|| VectorError::NotFound {
                    entity: "record".into(),
                    id: id.to_string(),
                })?;
        Ok(internal_id as usize)
    }

    /// Resolve a collection-scoped internal id to its UUID.
    pub async fn internal_to_uuid(
        &self,
        collection: &str,
        internal_id: usize,
    ) -> VectorResult<Uuid> {
        let id = sqlx::query_scalar::<_, String>(
            "SELECT id FROM vector_records WHERE collection = ? AND internal_id = ?",
        )
        .bind(collection)
        .bind(internal_id as i64)
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| VectorError::NotFound {
            entity: "record".into(),
            id: format!("{collection}:{internal_id}"),
        })?;
        Uuid::parse_str(&id)
            .map_err(|err| VectorError::Index(format!("invalid UUID stored in SQLite: {err}")))
    }

    /// Bulk-resolve collection-scoped internal ids to stored vector metadata.
    pub async fn bulk_internal_to_uuid(
        &self,
        collection: &str,
        ids: &[usize],
    ) -> VectorResult<Vec<(usize, VectorRecord)>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut builder = QueryBuilder::<Sqlite>::new(
            "SELECT id, internal_id, collection, text, metadata, created_at FROM vector_records WHERE collection = ",
        );
        builder.push_bind(collection);
        builder.push(" AND internal_id IN (");
        let mut separated = builder.separated(", ");
        for id in ids {
            separated.push_bind(*id as i64);
        }
        separated.push_unseparated(") ORDER BY internal_id ASC");

        let rows = builder
            .build_query_as::<RecordRow>()
            .fetch_all(&self.pool)
            .await?;

        let resolved = rows
            .into_iter()
            .map(record_from_row)
            .collect::<VectorResult<Vec<_>>>()?;

        let mut by_id = HashMap::with_capacity(resolved.len());
        for (record, internal_id) in resolved {
            by_id.insert(internal_id, record);
        }

        Ok(ids
            .iter()
            .filter_map(|id| by_id.remove(id).map(|record| (*id, record)))
            .collect())
    }

    /// Increment a collection's stored vector count.
    pub async fn increment_vector_count(&self, collection: &str, delta: i64) -> VectorResult<()> {
        sqlx::query(
            "UPDATE collections SET vector_count = MAX(vector_count + ?, 0) WHERE name = ?",
        )
        .bind(delta)
        .bind(collection)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Update the persisted index type for a collection.
    pub async fn update_collection_index_type(
        &self,
        collection: &str,
        index_type: IndexType,
    ) -> VectorResult<()> {
        sqlx::query("UPDATE collections SET index_type = ? WHERE name = ?")
            .bind(index_type_to_db(index_type))
            .bind(collection)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Return collection storage statistics as tracked in SQLite.
    pub async fn collection_stats(&self, name: &str) -> VectorResult<CollectionStats> {
        let vector_count =
            sqlx::query_scalar::<_, i64>("SELECT vector_count FROM collections WHERE name = ?")
                .bind(name)
                .fetch_optional(&self.pool)
                .await?
                .ok_or_else(|| VectorError::NotFound {
                    entity: "collection".into(),
                    id: name.to_string(),
                })?;

        let record_bytes = sqlx::query_scalar::<_, i64>(
            "SELECT COALESCE(SUM(LENGTH(id) + LENGTH(IFNULL(text, '')) + LENGTH(metadata) + LENGTH(created_at) + 8), 0) FROM vector_records WHERE collection = ?",
        )
        .bind(name)
        .fetch_one(&self.pool)
        .await?;

        let collection_bytes = sqlx::query_scalar::<_, i64>(
            "SELECT LENGTH(name) + LENGTH(distance) + LENGTH(index_type) + LENGTH(created_at) + LENGTH(metadata) + 32 FROM collections WHERE name = ?",
        )
        .bind(name)
        .fetch_one(&self.pool)
        .await?;

        Ok(CollectionStats {
            vector_count: vector_count as u64,
            size_bytes: (record_bytes + collection_bytes.max(0)) as u64,
        })
    }

    /// Return the next available internal id for a collection.
    pub async fn next_internal_id(&self, collection: &str) -> VectorResult<usize> {
        let max_internal_id = sqlx::query_scalar::<_, Option<i64>>(
            "SELECT MAX(internal_id) FROM vector_records WHERE collection = ?",
        )
        .bind(collection)
        .fetch_one(&self.pool)
        .await?;
        Ok(max_internal_id.map(|value| value as usize + 1).unwrap_or(0))
    }

    /// Load all persisted records for a collection, ordered by internal id.
    pub async fn list_records_for_collection(
        &self,
        collection: &str,
    ) -> VectorResult<Vec<(VectorRecord, usize)>> {
        let rows = sqlx::query_as::<_, RecordRow>(
            "SELECT id, internal_id, collection, text, metadata, created_at FROM vector_records WHERE collection = ? ORDER BY internal_id ASC",
        )
        .bind(collection)
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter().map(record_from_row).collect()
    }

    /// Search full-text records for a collection using SQLite FTS5.
    pub async fn keyword_search(
        &self,
        collection: &str,
        query: &str,
        limit: usize,
    ) -> VectorResult<Vec<(usize, VectorRecord, f32)>> {
        if query.trim().is_empty() || limit == 0 {
            return Ok(Vec::new());
        }

        let rows = sqlx::query_as::<_, KeywordRow>(
            r#"
            SELECT vr.id, vr.internal_id, vr.collection, vr.text, vr.metadata, vr.created_at,
                   CAST(bm25(vector_records_fts) AS REAL) AS rank
            FROM vector_records_fts
            JOIN vector_records AS vr ON vr.rowid = vector_records_fts.rowid
            WHERE vr.collection = ? AND vector_records_fts MATCH ?
            ORDER BY rank ASC
            LIMIT ?
            "#,
        )
        .bind(collection)
        .bind(query)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        rows.into_iter()
            .map(|row| {
                let rank = row.rank.unwrap_or(0.0);
                let record_row = RecordRow {
                    id: row.id,
                    internal_id: row.internal_id,
                    collection: row.collection,
                    text: row.text,
                    metadata: row.metadata,
                    created_at: row.created_at,
                };
                let (record, internal_id) = record_from_row(record_row)?;
                Ok((internal_id, record, rank))
            })
            .collect()
    }

    /// Close the underlying SQLx pool.
    pub async fn close(&self) {
        self.pool.close().await;
    }
}

#[derive(Debug, sqlx::FromRow)]
struct CollectionRow {
    name: String,
    dimensions: i64,
    distance: String,
    index_type: String,
    ef_construction: i64,
    m_connections: i64,
    created_at: String,
    vector_count: i64,
    metadata: String,
}

#[derive(Debug, sqlx::FromRow)]
struct RecordRow {
    id: String,
    internal_id: i64,
    collection: String,
    text: Option<String>,
    metadata: String,
    created_at: String,
}

#[derive(Debug, sqlx::FromRow)]
struct KeywordRow {
    id: String,
    internal_id: i64,
    collection: String,
    text: Option<String>,
    metadata: String,
    created_at: String,
    rank: Option<f32>,
}

/// Convert a raw database row into a [`Collection`], parsing JSON and RFC-3339 fields.
fn collection_from_row(row: CollectionRow) -> VectorResult<Collection> {
    Ok(Collection {
        name: row.name,
        dimensions: row.dimensions as usize,
        distance: distance_from_db(&row.distance)?,
        index_type: index_type_from_db(&row.index_type)?,
        ef_construction: row.ef_construction as usize,
        m_connections: row.m_connections as usize,
        created_at: chrono::DateTime::parse_from_rfc3339(&row.created_at)
            .map_err(|e| VectorError::Index(format!("invalid timestamp in DB: {e}")))?
            .with_timezone(&chrono::Utc),
        vector_count: row.vector_count as u64,
        metadata: parse_metadata(&row.metadata)?,
    })
}

fn record_from_row(row: RecordRow) -> VectorResult<(VectorRecord, usize)> {
    let id = Uuid::parse_str(&row.id).map_err(|err| {
        VectorError::Index(format!(
            "invalid UUID stored in vector_records table: {err}"
        ))
    })?;
    let record = VectorRecord {
        id,
        collection: row.collection,
        vector: Vec::new(),
        metadata: parse_metadata(&row.metadata)?,
        text: row.text,
        created_at: chrono::DateTime::parse_from_rfc3339(&row.created_at)
            .map_err(|e| VectorError::Index(format!("invalid timestamp in DB: {e}")))?
            .with_timezone(&chrono::Utc),
    };
    Ok((record, row.internal_id as usize))
}

fn normalize_metadata(metadata: &serde_json::Value) -> VectorResult<String> {
    if metadata.is_null() {
        Ok("{}".to_string())
    } else {
        serde_json::to_string(metadata).map_err(Into::into)
    }
}

fn parse_metadata(metadata: &str) -> VectorResult<serde_json::Value> {
    if metadata.trim().is_empty() {
        Ok(serde_json::json!({}))
    } else {
        Ok(serde_json::from_str(metadata)?)
    }
}

fn distance_to_db(distance: DistanceMetric) -> &'static str {
    match distance {
        DistanceMetric::Cosine => "cosine",
        DistanceMetric::Euclidean => "euclidean",
        DistanceMetric::DotProduct => "dot_product",
    }
}

fn distance_from_db(distance: &str) -> VectorResult<DistanceMetric> {
    match distance.trim_matches('"') {
        "cosine" => Ok(DistanceMetric::Cosine),
        "euclidean" => Ok(DistanceMetric::Euclidean),
        "dot_product" => Ok(DistanceMetric::DotProduct),
        other => Err(VectorError::Index(format!(
            "unsupported distance metric '{other}'"
        ))),
    }
}

fn index_type_to_db(index_type: IndexType) -> &'static str {
    match index_type {
        IndexType::HNSW => "hnsw",
        IndexType::Flat => "flat",
    }
}

fn index_type_from_db(index_type: &str) -> VectorResult<IndexType> {
    match index_type.trim_matches('"') {
        "hnsw" => Ok(IndexType::HNSW),
        "flat" => Ok(IndexType::Flat),
        other => Err(VectorError::Index(format!(
            "unsupported index type '{other}'"
        ))),
    }
}
