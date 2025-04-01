-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the jobs table
CREATE TABLE IF NOT EXISTS jobs_dev (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_title TEXT NOT NULL,
    job_url TEXT NOT NULL,
    positions_available INTEGER NOT NULL,
    hiring_urgency TEXT NOT NULL,
    seniority TEXT NOT NULL,
    work_arrangement TEXT[] NOT NULL,
    location_city TEXT[] NOT NULL,
    location_state TEXT[] NOT NULL,
    location_country TEXT NOT NULL,
    visa_sponsorship BOOLEAN NOT NULL,
    work_authorization TEXT NOT NULL,
    salary_range_min INTEGER NOT NULL,
    salary_range_max INTEGER NOT NULL,
    equity_range_min TEXT NOT NULL,
    equity_range_max TEXT NOT NULL,
    reporting_structure TEXT NOT NULL,
    team_structure TEXT NOT NULL,
    team_roles TEXT[] NOT NULL,
    role_status TEXT NOT NULL,
    role_category TEXT NOT NULL,
    tech_stack_must_haves TEXT[] NOT NULL,
    tech_stack_nice_to_haves TEXT[] NOT NULL,
    tech_stack_tags TEXT[] NOT NULL,
    tech_breadth TEXT NOT NULL,
    tech_breadth_requirement TEXT NOT NULL,
    minimum_years_of_experience INTEGER NOT NULL,
    domain_expertise TEXT[] NOT NULL,
    ai_ml_exp_required TEXT NOT NULL,
    ai_ml_exp_focus TEXT[] NOT NULL,
    infrastructure_experience TEXT[] NOT NULL,
    system_design_expectation TEXT NOT NULL,
    coding_proficiency TEXT NOT NULL,
    languages TEXT[] NOT NULL,
    version_control TEXT[] NOT NULL,
    ci_cd_tools TEXT[] NOT NULL,
    collaboration_tools TEXT[] NOT NULL,
    leadership_required BOOLEAN NOT NULL,
    education_required TEXT NOT NULL,
    education_advanced_degree TEXT NOT NULL,
    prior_startup_experience BOOLEAN NOT NULL,
    startup_exp TEXT NOT NULL,
    advancement_history_required BOOLEAN NOT NULL,
    career_trajectory TEXT NOT NULL,
    independent_work_capacity TEXT NOT NULL,
    independent_work TEXT NOT NULL,
    skills_must_have TEXT[] NOT NULL,
    skills_preferred TEXT[] NOT NULL,
    product_description TEXT NOT NULL,
    product_stage TEXT NOT NULL,
    product_dev_methodology TEXT[] NOT NULL,
    product_technical_challenges TEXT[] NOT NULL,
    product_development_stage TEXT NOT NULL,
    product_development_methodology TEXT[] NOT NULL,
    key_responsibilities TEXT[] NOT NULL,
    scope_of_impact TEXT[] NOT NULL,
    expected_deliverables TEXT[] NOT NULL,
    company_name TEXT NOT NULL,
    company_url TEXT NOT NULL,
    company_stage TEXT NOT NULL,
    company_funding_most_recent INTEGER NOT NULL,
    company_funding_total INTEGER NOT NULL,
    company_funding_investors TEXT[] NOT NULL,
    company_founded TEXT NOT NULL,
    company_team_size INTEGER NOT NULL,
    company_mission TEXT NOT NULL,
    company_vision TEXT NOT NULL,
    company_growth_story TEXT NOT NULL,
    company_culture TEXT NOT NULL,
    company_scaling_plans TEXT NOT NULL,
    company_mission_and_impact TEXT NOT NULL,
    company_tech_innovation TEXT NOT NULL,
    company_industry_vertical TEXT NOT NULL,
    company_target_market TEXT[] NOT NULL,
    ideal_companies TEXT[] NOT NULL,
    deal_breakers TEXT[] NOT NULL,
    disqualifying_traits TEXT[] NOT NULL,
    culture_fit TEXT[] NOT NULL,
    startup_mindset TEXT[] NOT NULL,
    autonomy_level_required TEXT NOT NULL,
    growth_mindset TEXT NOT NULL,
    ideal_candidate_profile TEXT NOT NULL,
    interview_process_steps TEXT[] NOT NULL,
    decision_makers TEXT[] NOT NULL,
    recruiter_pitch_points TEXT[] NOT NULL,
    embedding vector(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Drop existing function if it exists
DROP FUNCTION IF EXISTS match_jobs(vector(1536), float, int);

-- Create a function to match jobs based on embedding similarity
CREATE OR REPLACE FUNCTION match_jobs(
    query_embedding vector(1536),
    match_threshold float,
    match_count int
)
RETURNS TABLE (
    id UUID,
    job_title TEXT,
    company_name TEXT,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        j.id,
        j.job_title,
        j.company_name,
        1 - (j.embedding <=> query_embedding) as similarity
    FROM jobs_dev j
    WHERE 1 - (j.embedding <=> query_embedding) > match_threshold
    ORDER BY j.embedding <=> query_embedding
    LIMIT match_count;
END;
$$; 