-- First, make the columns nullable
ALTER TABLE jobs_dev ALTER COLUMN equity_range_min DROP NOT NULL;
ALTER TABLE jobs_dev ALTER COLUMN equity_range_max DROP NOT NULL;

-- Convert the columns to text first to handle 'n/a' values
ALTER TABLE jobs_dev ALTER COLUMN equity_range_min TYPE text;
ALTER TABLE jobs_dev ALTER COLUMN equity_range_max TYPE text;

-- Update existing 'n/a' values to NULL
UPDATE jobs_dev 
SET equity_range_min = NULL 
WHERE equity_range_min = 'n/a';

UPDATE jobs_dev 
SET equity_range_max = NULL 
WHERE equity_range_max = 'n/a';

-- Finally, convert to float, handling NULL values
ALTER TABLE jobs_dev ALTER COLUMN equity_range_min TYPE float USING CASE 
    WHEN equity_range_min IS NULL THEN NULL 
    ELSE equity_range_min::float 
END;
ALTER TABLE jobs_dev ALTER COLUMN equity_range_max TYPE float USING CASE 
    WHEN equity_range_max IS NULL THEN NULL 
    ELSE equity_range_max::float 
END; 