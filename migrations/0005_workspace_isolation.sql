PRAGMA foreign_keys = OFF;

ALTER TABLE collections RENAME TO collections_old;

CREATE TABLE collections (
    workspace_id     TEXT NOT NULL,
    name             TEXT NOT NULL,
    dimensions       INTEGER NOT NULL,
    distance         TEXT NOT NULL,
    index_type       TEXT NOT NULL,
    ef_construction  INTEGER NOT NULL DEFAULT 200,
    m_connections    INTEGER NOT NULL DEFAULT 16,
    created_at       TEXT NOT NULL,
    vector_count     INTEGER NOT NULL DEFAULT 0,
    metadata         TEXT NOT NULL DEFAULT '{}',
    UNIQUE(workspace_id, name)
);

INSERT INTO collections (
    workspace_id,
    name,
    dimensions,
    distance,
    index_type,
    ef_construction,
    m_connections,
    created_at,
    vector_count,
    metadata
)
SELECT
    'default',
    name,
    dimensions,
    distance,
    index_type,
    ef_construction,
    m_connections,
    created_at,
    vector_count,
    metadata
FROM collections_old;

ALTER TABLE vector_records RENAME TO vector_records_old;

CREATE TABLE vector_records (
    id           TEXT PRIMARY KEY NOT NULL,
    internal_id  INTEGER NOT NULL,
    workspace_id TEXT NOT NULL,
    collection   TEXT NOT NULL,
    text         TEXT,
    metadata     TEXT NOT NULL DEFAULT '{}',
    created_at   TEXT NOT NULL,
    FOREIGN KEY (workspace_id, collection) REFERENCES collections(workspace_id, name)
);

INSERT INTO vector_records (
    id,
    internal_id,
    workspace_id,
    collection,
    text,
    metadata,
    created_at
)
SELECT
    id,
    internal_id,
    'default',
    collection,
    text,
    metadata,
    created_at
FROM vector_records_old;

DROP TABLE collections_old;
DROP TABLE vector_records_old;

CREATE UNIQUE INDEX IF NOT EXISTS idx_vector_records_workspace_collection_internal
    ON vector_records(workspace_id, collection, internal_id);
CREATE INDEX IF NOT EXISTS idx_vector_records_workspace_collection
    ON vector_records(workspace_id, collection);

DROP TRIGGER IF EXISTS vector_records_ai;
DROP TRIGGER IF EXISTS vector_records_ad;
DROP TRIGGER IF EXISTS vector_records_au;
DROP TABLE IF EXISTS vector_records_fts;

CREATE VIRTUAL TABLE vector_records_fts USING fts5(
    text,
    metadata,
    content='vector_records',
    content_rowid='rowid'
);

INSERT INTO vector_records_fts(rowid, text, metadata)
SELECT rowid, COALESCE(text, ''), metadata
FROM vector_records;

CREATE TRIGGER vector_records_ai AFTER INSERT ON vector_records BEGIN
    INSERT INTO vector_records_fts(rowid, text, metadata)
    VALUES (new.rowid, COALESCE(new.text, ''), new.metadata);
END;

CREATE TRIGGER vector_records_ad AFTER DELETE ON vector_records BEGIN
    INSERT INTO vector_records_fts(vector_records_fts, rowid, text, metadata)
    VALUES ('delete', old.rowid, COALESCE(old.text, ''), old.metadata);
END;

CREATE TRIGGER vector_records_au AFTER UPDATE ON vector_records BEGIN
    INSERT INTO vector_records_fts(vector_records_fts, rowid, text, metadata)
    VALUES ('delete', old.rowid, COALESCE(old.text, ''), old.metadata);
    INSERT INTO vector_records_fts(rowid, text, metadata)
    VALUES (new.rowid, COALESCE(new.text, ''), new.metadata);
END;

PRAGMA foreign_keys = ON;
