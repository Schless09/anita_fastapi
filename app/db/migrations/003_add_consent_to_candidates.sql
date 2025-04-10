-- Add sms_consent and legal_consent columns to the candidates_dev table

ALTER TABLE candidates_dev
ADD COLUMN IF NOT EXISTS sms_consent BOOLEAN NOT NULL DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS legal_consent BOOLEAN NOT NULL DEFAULT FALSE;

-- It's often a good idea to add these to the production table as well, 
-- but apply migrations carefully in production environments.
-- Uncomment the following lines if you want to apply to candidates_prod too.
-- ALTER TABLE candidates_prod
-- ADD COLUMN IF NOT EXISTS sms_consent BOOLEAN NOT NULL DEFAULT FALSE,
-- ADD COLUMN IF NOT EXISTS legal_consent BOOLEAN NOT NULL DEFAULT FALSE; 