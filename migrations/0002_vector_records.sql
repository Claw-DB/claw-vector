CREATE TABLE IF NOT EXISTS vector_records (
    id          TEXT PRIMARY KEY NOT NULL,
    internal_id INTEGER NOT NULL,
    collection  TEXT NOT NULL,
    text        TEXT,
    metadata    TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL,
    FOREIGN KEY (collection) REFERENCES collections(name)
);
