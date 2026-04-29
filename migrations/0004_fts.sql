CREATE VIRTUAL TABLE IF NOT EXISTS vector_records_fts USING fts5(
    text,
    metadata,
    content='vector_records',
    content_rowid='rowid'
);

INSERT INTO vector_records_fts(rowid, text, metadata)
SELECT rowid, COALESCE(text, ''), metadata
FROM vector_records
WHERE NOT EXISTS (
    SELECT 1 FROM vector_records_fts WHERE vector_records_fts.rowid = vector_records.rowid
);

CREATE TRIGGER IF NOT EXISTS vector_records_ai AFTER INSERT ON vector_records BEGIN
    INSERT INTO vector_records_fts(rowid, text, metadata)
    VALUES (new.rowid, COALESCE(new.text, ''), new.metadata);
END;

CREATE TRIGGER IF NOT EXISTS vector_records_ad AFTER DELETE ON vector_records BEGIN
    INSERT INTO vector_records_fts(vector_records_fts, rowid, text, metadata)
    VALUES ('delete', old.rowid, COALESCE(old.text, ''), old.metadata);
END;

CREATE TRIGGER IF NOT EXISTS vector_records_au AFTER UPDATE ON vector_records BEGIN
    INSERT INTO vector_records_fts(vector_records_fts, rowid, text, metadata)
    VALUES ('delete', old.rowid, COALESCE(old.text, ''), old.metadata);
    INSERT INTO vector_records_fts(rowid, text, metadata)
    VALUES (new.rowid, COALESCE(new.text, ''), new.metadata);
END;
