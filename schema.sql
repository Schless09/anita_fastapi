-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create calls-dev table
CREATE TABLE IF NOT EXISTS "calls-dev" (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    candidate_id UUID NOT NULL,
    call_id TEXT NOT NULL,
    transcript TEXT,
    summary TEXT,
    analysis JSONB,
    email_sent BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for calls-dev
CREATE INDEX IF NOT EXISTS idx_calls_dev_candidate_id ON "calls-dev" (candidate_id);
CREATE INDEX IF NOT EXISTS idx_calls_dev_call_id ON "calls-dev" (call_id);

-- Add trigger for updated_at
DROP TRIGGER IF EXISTS update_calls_updated_at ON "calls-dev";
CREATE TRIGGER update_calls_updated_at
BEFORE UPDATE ON "calls-dev"
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column(); 