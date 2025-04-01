-- Create the job_listings table if it doesn't exist
CREATE TABLE IF NOT EXISTS job_listings (
    id uuid DEFAULT uuid_generate_v4() PRIMARY KEY,
    created_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL,
    
    -- Job Details
    job_title text NOT NULL,
    job_url text,
    positions_available text,
    hiring_urgency text,
    seniority seniority NOT NULL,
    work_arrangement work_arrangement[] NOT NULL,
    location jsonb NOT NULL,
    visa_sponsorship boolean DEFAULT false,
    work_authorization text,
    
    -- Compensation
    compensation jsonb NOT NULL,
    reporting_structure text,
    team_composition jsonb NOT NULL,
    
    -- Role Details
    role_status role_status NOT NULL,
    role_category role_category NOT NULL,
    tech_stack jsonb NOT NULL,
    tech_breadth tech_breadth NOT NULL,
    minimum_years_of_experience text,
    domain_expertise text[] DEFAULT '{}',
    
    -- AI/ML Experience
    ai_ml_experience jsonb,
    infrastructure_experience text[] DEFAULT '{}',
    system_design_expectation text,
    coding_proficiency coding_proficiency NOT NULL,
    
    -- Skills & Requirements
    languages text[] DEFAULT '{}',
    version_control text[] DEFAULT '{}',
    ci_cd_tools text[] DEFAULT '{}',
    collaboration_tools text[] DEFAULT '{}',
    leadership_required boolean DEFAULT false,
    education_requirements jsonb,
    prior_startup_experience boolean DEFAULT false,
    advancement_history_required boolean DEFAULT false,
    independent_work_capacity autonomy_level,
    skills_must_have text[] DEFAULT '{}',
    skills_preferred text[] DEFAULT '{}',
    
    -- Product Info
    product jsonb NOT NULL,
    key_responsibilities text[] DEFAULT '{}',
    scope_of_impact text[] DEFAULT '{}',
    expected_deliverables text[] DEFAULT '{}',
    
    -- Company Info
    company jsonb NOT NULL,
    
    -- Candidate Fit
    candidate_fit jsonb NOT NULL,
    
    -- Process & Pitch
    interview_process jsonb,
    recruiter_pitch_points text[] DEFAULT '{}'
);

-- Add updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = timezone('utc'::text, now());
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_job_listings_updated_at
    BEFORE UPDATE ON job_listings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Enable RLS
ALTER TABLE job_listings ENABLE ROW LEVEL SECURITY;

-- Create policies
CREATE POLICY "Enable read access for all users" ON job_listings FOR SELECT USING (true);
CREATE POLICY "Enable insert for authenticated users only" ON job_listings FOR INSERT WITH CHECK (auth.role() = 'authenticated');
CREATE POLICY "Enable update for authenticated users only" ON job_listings FOR UPDATE USING (auth.role() = 'authenticated');
CREATE POLICY "Enable delete for authenticated users only" ON job_listings FOR DELETE USING (auth.role() = 'authenticated'); 