-- Create enum types for the various categorical fields
CREATE TYPE company_stage AS ENUM (
    'Pre-Seed', 'Seed', 'Series A', 'Series B', 'Series C+', 'Private', 'Public'
);

CREATE TYPE company_industry_vertical AS ENUM (
    'AI', 'Healthcare', 'Fintech', 'Insurtech', 'Enterprise Software', 'Developer Tools',
    'Cybersecurity', 'Media', 'Construction Tech', 'E-commerce', 'Logistics', 'Robotics',
    'Climate', 'Education', 'LegalTech', 'Biotech', 'IoT', 'Consumer', 'Real Estate',
    'HR Tech', 'Gaming', 'Travel', 'Supply Chain', 'Manufacturing'
);

CREATE TYPE target_market AS ENUM (
    'B2B', 'B2C', 'Enterprise', 'SMB', 'Consumer'
);

CREATE TYPE role_category AS ENUM (
    'Frontend', 'Backend', 'Full-Stack', 'Infra', 'DevOps', 'Data', 'ML/AI', 'Mobile',
    'SWE', 'Design', 'Product', 'Security', 'Founding Engineer', 'QA', 'Embedded'
);

CREATE TYPE seniority AS ENUM (
    '0-3', '3-5', '5-8', '8+'
);

CREATE TYPE work_arrangement AS ENUM (
    'On-site', 'Hybrid', 'Remote'
);

CREATE TYPE role_status AS ENUM (
    'Active', 'Inactive', 'Closed'
);

CREATE TYPE autonomy_level AS ENUM (
    'Low', 'Medium', 'High', 'Very High'
);

CREATE TYPE coding_proficiency AS ENUM (
    'Basic', 'Intermediate', 'Advanced', 'Expert'
);

CREATE TYPE tech_breadth AS ENUM (
    'Frontend', 'Backend', 'Full-Stack', 'Infra', 'AI/ML'
);

CREATE TYPE requirement_level AS ENUM (
    'Required', 'Preferred', 'Not Required'
);

CREATE TYPE product_stage AS ENUM (
    'Prototype', 'MVP', 'Market-ready', 'Scaling', 'Established'
);

CREATE TYPE dev_methodology AS ENUM (
    'Agile', 'Scrum', 'Kanban', 'Hacker', 'Owner', 'n/a'
);

CREATE TYPE education_requirement AS ENUM (
    'Top 30 CS program', 'Ivy League', 'Respected schools', 'No requirement', 'No bootcamps'
);

CREATE TYPE advanced_degree AS ENUM (
    'PhD preferred', 'Master''s preferred', 'Not required'
);

-- Create the main job_listings table
CREATE TABLE job_listings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Basic job info
    job_title TEXT NOT NULL,
    job_url TEXT NOT NULL,
    positions_available INTEGER NOT NULL,
    hiring_urgency TEXT NOT NULL,
    seniority seniority NOT NULL,
    seniority_level seniority,
    work_arrangement work_arrangement[] NOT NULL,
    
    -- Location
    location_city TEXT[] NOT NULL,
    location_state TEXT[] NOT NULL,
    location_country TEXT NOT NULL,
    
    -- Visa and work authorization
    visa_sponsorship BOOLEAN NOT NULL,
    work_authorization TEXT NOT NULL,
    
    -- Compensation
    salary_range_min INTEGER NOT NULL,
    salary_range_max INTEGER NOT NULL,
    equity_range_min TEXT NOT NULL,
    equity_range_max TEXT NOT NULL,
    
    -- Reporting and team
    reporting_structure TEXT NOT NULL,
    team_size TEXT NOT NULL,
    team_structure TEXT NOT NULL,
    team_roles TEXT[] NOT NULL,
    
    -- Role details
    role_status role_status NOT NULL,
    role_category role_category NOT NULL,
    tech_stack_must_haves TEXT[] NOT NULL,
    tech_stack_nice_to_haves TEXT[] NOT NULL,
    tech_stack_tags TEXT[] DEFAULT '{}',
    tech_breadth tech_breadth NOT NULL,
    tech_breadth_requirement tech_breadth,
    minimum_years_of_experience INTEGER NOT NULL,
    domain_expertise TEXT[] NOT NULL,
    
    -- AI/ML experience
    ai_ml_exp_required requirement_level NOT NULL,
    ai_ml_exp_focus TEXT[] NOT NULL,
    
    -- Infrastructure
    infrastructure_experience TEXT[] NOT NULL,
    system_design_expectation TEXT NOT NULL,
    coding_proficiency coding_proficiency NOT NULL,
    languages TEXT[] NOT NULL,
    version_control TEXT[] NOT NULL,
    ci_cd_tools TEXT[] NOT NULL,
    collaboration_tools TEXT[] NOT NULL,
    
    -- Requirements
    leadership_required BOOLEAN NOT NULL,
    education_required education_requirement NOT NULL,
    education_advanced_degree advanced_degree NOT NULL,
    prior_startup_experience BOOLEAN NOT NULL,
    startup_exp requirement_level NOT NULL,
    advancement_history_required BOOLEAN NOT NULL,
    career_trajectory requirement_level NOT NULL,
    independent_work_capacity autonomy_level,
    independent_work autonomy_level NOT NULL,
    
    -- Skills
    skills_must_have TEXT[] NOT NULL,
    skills_preferred TEXT[] NOT NULL,
    
    -- Product info
    product_description TEXT NOT NULL,
    product_stage product_stage NOT NULL,
    product_dev_methodology dev_methodology[] NOT NULL,
    product_technical_challenges TEXT[] NOT NULL,
    product_development_stage product_stage,
    product_development_methodology dev_methodology[],
    
    -- Responsibilities and deliverables
    key_responsibilities TEXT[] NOT NULL,
    scope_of_impact TEXT[] NOT NULL,
    expected_deliverables TEXT[] NOT NULL,
    
    -- Company info
    company_name TEXT NOT NULL,
    company_url TEXT NOT NULL,
    company_stage company_stage NOT NULL,
    company_funding_most_recent INTEGER NOT NULL,
    company_funding_total INTEGER NOT NULL,
    company_funding_investors TEXT[] NOT NULL,
    company_team_size TEXT NOT NULL,
    company_founded TEXT NOT NULL,
    company_mission TEXT NOT NULL,
    company_vision TEXT NOT NULL,
    company_growth_story TEXT NOT NULL,
    company_culture TEXT NOT NULL,
    company_scaling_plans TEXT NOT NULL,
    company_mission_and_impact TEXT NOT NULL,
    company_tech_innovation TEXT NOT NULL,
    company_industry_vertical company_industry_vertical,
    company_target_market target_market[],
    
    -- Candidate fit
    ideal_companies TEXT[] NOT NULL,
    deal_breakers TEXT[] NOT NULL,
    disqualifying_traits TEXT[] NOT NULL,
    culture_fit TEXT[] NOT NULL,
    startup_mindset TEXT[] NOT NULL,
    autonomy_level_required autonomy_level NOT NULL,
    growth_mindset TEXT NOT NULL,
    ideal_candidate_profile TEXT,
    
    -- Process and pitch
    interview_process_description TEXT NOT NULL,
    interview_process_steps TEXT[],
    interview_process_duration TEXT,
    interview_process_format TEXT,
    interview_process_tags TEXT[] DEFAULT '{}',
    interview_process_assessment_type TEXT,
    interview_process_focus_areas TEXT[] DEFAULT '{}',
    interview_process_time_to_hire TEXT,
    interview_process_decision_makers TEXT[] DEFAULT '{}',
    recruiter_pitch_points TEXT[] NOT NULL,
    
    -- Embeddings for semantic search
    embedding vector(1536)
);

-- Create an index on the embedding column for faster similarity search
CREATE INDEX job_listings_embedding_idx ON job_listings USING ivfflat (embedding vector_cosine_ops);

-- Create a function to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create a trigger to automatically update the updated_at column
CREATE TRIGGER update_job_listings_updated_at
    BEFORE UPDATE ON job_listings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create RLS policies
ALTER TABLE job_listings ENABLE ROW LEVEL SECURITY;

-- Allow public read access
CREATE POLICY "Allow public read access"
    ON job_listings FOR SELECT
    TO public
    USING (true);

-- Allow authenticated users to insert
CREATE POLICY "Allow authenticated users to insert"
    ON job_listings FOR INSERT
    TO authenticated
    WITH CHECK (true);

-- Allow authenticated users to update their own records
CREATE POLICY "Allow authenticated users to update"
    ON job_listings FOR UPDATE
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- Allow authenticated users to delete their own records
CREATE POLICY "Allow authenticated users to delete"
    ON job_listings FOR DELETE
    TO authenticated
    USING (true)
    WITH CHECK (true); 