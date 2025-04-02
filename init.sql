-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Tạo bảng 'images'
CREATE TABLE IF NOT EXISTS images (
    id SERIAL PRIMARY KEY,
    filename VARCHAR NOT NULL,
    content_type VARCHAR NOT NULL,
    file_size INTEGER,
    storage_path VARCHAR NOT NULL,
    is_validate BOOLEAN NOT NULL DEFAULT FALSE
);

-- Tạo bảng 'users'
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL
);

-- Tạo bảng 'embeddings'
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    vector VECTOR(512),
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    image_id INTEGER REFERENCES images(id) ON DELETE CASCADE
);


-- Import CSV data into tables
COPY users FROM 'data/users.csv' DELIMITER ',' CSV HEADER;
COPY embeddings FROM 'data/embeddings.csv' DELIMITER ',' CSV HEADER;
COPY images FROM 'data/images.csv' DELIMITER ',' CSV HEADER;
