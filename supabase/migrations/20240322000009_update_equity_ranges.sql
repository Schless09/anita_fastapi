-- First, create temporary columns to store the values
ALTER TABLE jobs_dev ADD COLUMN equity_range_min_new float;
ALTER TABLE jobs_dev ADD COLUMN equity_range_max_new float;

-- Convert existing values to float, handling 'n/a' and percentage values
UPDATE jobs_dev 
SET equity_range_min = CASE 
    WHEN equity_range_min::text = 'n/a' OR equity_range_min IS NULL THEN NULL 
    ELSE REPLACE(equity_range_min::text, '%', '')::float 
END,
equity_range_max = CASE 
    WHEN equity_range_max::text = 'n/a' OR equity_range_max IS NULL THEN NULL 
    ELSE REPLACE(equity_range_max::text, '%', '')::float 
END;

-- Drop the old columns
ALTER TABLE jobs_dev DROP COLUMN equity_range_min;
ALTER TABLE jobs_dev DROP COLUMN equity_range_max;

-- Rename the new columns to the original names
ALTER TABLE jobs_dev RENAME COLUMN equity_range_min_new TO equity_range_min;
ALTER TABLE jobs_dev RENAME COLUMN equity_range_max_new TO equity_range_max;

-- Make the columns nullable since we allow 'n/a' values
ALTER TABLE jobs_dev ALTER COLUMN equity_range_min DROP NOT NULL;
ALTER TABLE jobs_dev ALTER COLUMN equity_range_max DROP NOT NULL; 