-- First, add the new column for career_trajectory
ALTER TABLE jobs_dev ADD COLUMN IF NOT EXISTS career_trajectory text;

-- Copy existing data from advancement_history_required to career_trajectory
UPDATE jobs_dev SET career_trajectory = advancement_history_required;

-- Drop the old columns
ALTER TABLE jobs_dev DROP COLUMN IF EXISTS advancement_history_required;
ALTER TABLE jobs_dev DROP COLUMN IF EXISTS tech_breadth; 