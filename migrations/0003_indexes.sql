CREATE INDEX IF NOT EXISTS idx_vector_records_collection  ON vector_records(collection);
CREATE INDEX IF NOT EXISTS idx_vector_records_created_at  ON vector_records(created_at);
CREATE INDEX IF NOT EXISTS idx_vector_records_internal_id ON vector_records(internal_id);
