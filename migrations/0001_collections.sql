CREATE TABLE IF NOT EXISTS collections (
    name             TEXT PRIMARY KEY NOT NULL,
    dimensions       INTEGER NOT NULL,
    distance         TEXT NOT NULL,
    index_type       TEXT NOT NULL,
    ef_construction  INTEGER NOT NULL DEFAULT 200,
    m_connections    INTEGER NOT NULL DEFAULT 16,
    created_at       TEXT NOT NULL,
    vector_count     INTEGER NOT NULL DEFAULT 0,
    metadata         TEXT NOT NULL DEFAULT '{}'
);
