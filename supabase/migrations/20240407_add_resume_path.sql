-- Add resume_path column to candidates table
ALTER TABLE candidates_dev ADD COLUMN IF NOT EXISTS resume_path TEXT;
ALTER TABLE candidates_prod ADD COLUMN IF NOT EXISTS resume_path TEXT;

-- Add index for faster lookups
CREATE INDEX IF NOT EXISTS idx_candidates_dev_resume_path ON candidates_dev(resume_path);
CREATE INDEX IF NOT EXISTS idx_candidates_prod_resume_path ON candidates_prod(resume_path);

-- Comment on columns
COMMENT ON COLUMN candidates_dev.resume_path IS 'Path to the resume file in Supabase storage';
COMMENT ON COLUMN candidates_prod.resume_path IS 'Path to the resume file in Supabase storage'; 