-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Add embedding column to candidates table if it doesn't exist
ALTER TABLE candidates_dev ADD COLUMN IF NOT EXISTS embedding vector(1536);
ALTER TABLE candidates_dev ADD COLUMN IF NOT EXISTS embedding_metadata jsonb;

-- Add embedding column to jobs table if it doesn't exist
CREATE TABLE IF NOT EXISTS jobs_dev (
    id uuid PRIMARY KEY,
    title text,
    company text,
    description text,
    profile_json jsonb,
    embedding vector(1536),
    embedding_metadata jsonb,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);

-- Create function to match candidates based on vector similarity
CREATE OR REPLACE FUNCTION match_candidates(query_embedding vector, match_threshold float, match_count int)
RETURNS TABLE(
    id uuid,
    full_name text,
    email text,
    phone text,
    linkedin_url text,
    profile_json jsonb,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id,
        c.full_name,
        c.email,
        c.phone,
        c.linkedin_url,
        c.profile_json,
        1 - (c.embedding <=> query_embedding) as similarity
    FROM
        candidates_dev c
    WHERE
        c.embedding IS NOT NULL
    ORDER BY
        c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Create function to match jobs based on vector similarity
CREATE OR REPLACE FUNCTION match_jobs(query_embedding vector, match_threshold float, match_count int)
RETURNS TABLE(
    id bigint,
    job_title text,
    company_name text,
    key_responsibilities text[],
    skills_must_have text[],
    skills_preferred text[],
    tech_stack_must_haves text[],
    tech_stack_nice_to_haves text[],
    role_category text[],
    seniority text,
    scope_of_impact text[],
    company_mission text,
    company_vision text,
    company_culture text,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        j.id,
        j.title,
        j.company,
        j.key_responsibilities,
        j.skills_must_have,
        j.skills_preferred,
        j.tech_stack_must_haves,
        j.tech_stack_nice_to_haves,
        j.role_category,
        j.seniority,
        j.scope_of_impact,
        j.company_mission,
        j.company_vision,
        j.company_culture,
        1 - (j.embedding <=> query_embedding) as similarity
    FROM
        jobs_dev j
    WHERE
        j.embedding IS NOT NULL
    ORDER BY
        j.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Create indexes for faster vector search
CREATE INDEX IF NOT EXISTS candidates_embedding_idx ON candidates_dev USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS jobs_embedding_idx ON jobs_dev USING ivfflat (embedding vector_cosine_ops);

-- Function to update a candidate with embedding
CREATE OR REPLACE FUNCTION update_candidate_embedding(
    candidate_id uuid,
    embedding_vector vector,
    metadata jsonb
)
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE candidates_dev
    SET
        embedding = embedding_vector,
        embedding_metadata = metadata,
        updated_at = NOW()
    WHERE id = candidate_id;
END;
$$;

-- Function to update a job with embedding
CREATE OR REPLACE FUNCTION update_job_embedding(
    job_id uuid,
    embedding_vector vector,
    metadata jsonb
)
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE jobs_dev
    SET
        embedding = embedding_vector,
        embedding_metadata = metadata,
        updated_at = NOW()
    WHERE id = job_id;
END;
$$; 