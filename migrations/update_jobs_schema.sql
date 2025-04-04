-- Create jobs_dev table if it doesn't exist
CREATE TABLE IF NOT EXISTS jobs_dev (
    id uuid PRIMARY KEY,
    title text,
    company text,
    description text,
    profile_json jsonb,
    embedding vector(1536),
    embedding_metadata jsonb,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);

-- Add new columns to jobs_dev table
ALTER TABLE jobs_dev
ADD COLUMN IF NOT EXISTS job_title text,
ADD COLUMN IF NOT EXISTS job_url text,
ADD COLUMN IF NOT EXISTS positions_available text,
ADD COLUMN IF NOT EXISTS hiring_urgency text,
ADD COLUMN IF NOT EXISTS seniority_level text,
ADD COLUMN IF NOT EXISTS work_arrangement text[],
ADD COLUMN IF NOT EXISTS city text[],
ADD COLUMN IF NOT EXISTS state text[],
ADD COLUMN IF NOT EXISTS country text,
ADD COLUMN IF NOT EXISTS visa_sponsorship boolean,
ADD COLUMN IF NOT EXISTS work_authorization text,
ADD COLUMN IF NOT EXISTS salary_min text,
ADD COLUMN IF NOT EXISTS salary_max text,
ADD COLUMN IF NOT EXISTS equity_min text,
ADD COLUMN IF NOT EXISTS equity_max text,
ADD COLUMN IF NOT EXISTS reporting_structure text,
ADD COLUMN IF NOT EXISTS team_size text,
ADD COLUMN IF NOT EXISTS team_structure text,
ADD COLUMN IF NOT EXISTS team_roles text[],
ADD COLUMN IF NOT EXISTS role_status text,
ADD COLUMN IF NOT EXISTS role_category text,
ADD COLUMN IF NOT EXISTS tech_stack_must_haves text[],
ADD COLUMN IF NOT EXISTS tech_stack_nice_to_haves text[],
ADD COLUMN IF NOT EXISTS tech_stack_tags text[],
ADD COLUMN IF NOT EXISTS tech_breadth_requirement text,
ADD COLUMN IF NOT EXISTS minimum_years_of_experience text,
ADD COLUMN IF NOT EXISTS domain_expertise text[],
ADD COLUMN IF NOT EXISTS ai_ml_experience_required text,
ADD COLUMN IF NOT EXISTS infrastructure_experience text[],
ADD COLUMN IF NOT EXISTS system_design_expectation text,
ADD COLUMN IF NOT EXISTS coding_proficiency text,
ADD COLUMN IF NOT EXISTS languages text[],
ADD COLUMN IF NOT EXISTS version_control text[],
ADD COLUMN IF NOT EXISTS ci_cd_tools text[],
ADD COLUMN IF NOT EXISTS collaboration_tools text[],
ADD COLUMN IF NOT EXISTS leadership_required boolean,
ADD COLUMN IF NOT EXISTS education_required text,
ADD COLUMN IF NOT EXISTS advanced_degree_preferred text,
ADD COLUMN IF NOT EXISTS prior_startup_experience text,
ADD COLUMN IF NOT EXISTS advancement_history_required boolean,
ADD COLUMN IF NOT EXISTS independent_work_capacity text,
ADD COLUMN IF NOT EXISTS skills_must_have text[],
ADD COLUMN IF NOT EXISTS skills_preferred text[],
ADD COLUMN IF NOT EXISTS product_description text,
ADD COLUMN IF NOT EXISTS product_development_stage text,
ADD COLUMN IF NOT EXISTS product_technical_challenges text[],
ADD COLUMN IF NOT EXISTS product_development_methodology text[],
ADD COLUMN IF NOT EXISTS key_responsibilities text[],
ADD COLUMN IF NOT EXISTS scope_of_impact text[],
ADD COLUMN IF NOT EXISTS expected_deliverables text[],
ADD COLUMN IF NOT EXISTS ideal_companies text[],
ADD COLUMN IF NOT EXISTS deal_breakers text[],
ADD COLUMN IF NOT EXISTS disqualifying_traits text[],
ADD COLUMN IF NOT EXISTS culture_fit text[],
ADD COLUMN IF NOT EXISTS startup_mindset text[],
ADD COLUMN IF NOT EXISTS autonomy_level_required text,
ADD COLUMN IF NOT EXISTS growth_mindset text[],
ADD COLUMN IF NOT EXISTS ideal_candidate_profile text,
ADD COLUMN IF NOT EXISTS interview_process text,
ADD COLUMN IF NOT EXISTS recruiter_pitch_points text[],
ADD COLUMN IF NOT EXISTS company_id uuid; 