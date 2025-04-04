-- Add error_message column to candidates table
ALTER TABLE candidates_prod ADD COLUMN IF NOT EXISTS error_message text;
ALTER TABLE candidates_dev ADD COLUMN IF NOT EXISTS error_message text; 