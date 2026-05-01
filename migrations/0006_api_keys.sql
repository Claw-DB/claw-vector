CREATE TABLE IF NOT EXISTS api_keys (
    key_hash TEXT PRIMARY KEY NOT NULL,
    workspace_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    revoked INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_api_keys_workspace ON api_keys(workspace_id);
