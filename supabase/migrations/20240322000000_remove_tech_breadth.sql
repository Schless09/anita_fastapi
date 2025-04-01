-- First, alter the columns to use TEXT instead of the enum
ALTER TABLE job_listings 
    ALTER COLUMN tech_breadth TYPE TEXT,
    ALTER COLUMN tech_breadth_requirement TYPE TEXT;

-- Then drop the enum type
DROP TYPE tech_breadth; 