-- Drop existing enums if they exist
DO $$ BEGIN
    DROP TYPE IF EXISTS company_stage CASCADE;
    DROP TYPE IF EXISTS company_industry_vertical CASCADE;
    DROP TYPE IF EXISTS target_market CASCADE;
    DROP TYPE IF EXISTS role_category CASCADE;
    DROP TYPE IF EXISTS seniority CASCADE;
    DROP TYPE IF EXISTS work_arrangement CASCADE;
    DROP TYPE IF EXISTS role_status CASCADE;
    DROP TYPE IF EXISTS tech_breadth CASCADE;
    DROP TYPE IF EXISTS aiml_exp_required CASCADE;
    DROP TYPE IF EXISTS coding_proficiency CASCADE;
    DROP TYPE IF EXISTS education_required CASCADE;
    DROP TYPE IF EXISTS education_advanced_degree CASCADE;
    DROP TYPE IF EXISTS autonomy_level CASCADE;
EXCEPTION
    WHEN OTHERS THEN null;
END $$;

-- Create company_stage enum
CREATE TYPE company_stage AS ENUM (
    'Pre-Seed',
    'Seed',
    'Series A',
    'Series B',
    'Series C+',
    'Private',
    'Public'
);

-- Create company_industry_vertical enum
CREATE TYPE company_industry_vertical AS ENUM (
    'AI',
    'Healthcare',
    'Fintech',
    'Insurtech',
    'Enterprise Software',
    'Developer Tools',
    'Cybersecurity',
    'Media',
    'Construction Tech',
    'E-commerce',
    'Logistics',
    'Robotics',
    'Climate',
    'Education',
    'LegalTech',
    'Biotech',
    'IoT',
    'Consumer',
    'Real Estate',
    'HR Tech',
    'Gaming',
    'Travel',
    'Supply Chain',
    'Manufacturing'
);

-- Create target_market enum
CREATE TYPE target_market AS ENUM (
    'B2B',
    'B2C',
    'Enterprise',
    'SMB',
    'Consumer'
);

-- Create role_category enum
CREATE TYPE role_category AS ENUM (
    'Frontend',
    'Backend',
    'Full-Stack',
    'Infra',
    'DevOps',
    'Data',
    'ML/AI',
    'Mobile',
    'SWE',
    'Design',
    'Product',
    'Security',
    'Founding Engineer',
    'QA',
    'Embedded'
);

-- Create seniority enum
CREATE TYPE seniority AS ENUM (
    '0-3',
    '3-5',
    '5-8',
    '8+'
);

-- Create work_arrangement enum
CREATE TYPE work_arrangement AS ENUM (
    'On-site',
    'Hybrid',
    'Remote'
);

-- Create role_status enum
CREATE TYPE role_status AS ENUM (
    'Active',
    'Inactive',
    'Closed'
);

-- Create tech_breadth enum
CREATE TYPE tech_breadth AS ENUM (
    'Frontend',
    'Backend',
    'Full-Stack',
    'Infra',
    'AI/ML'
);

-- Create aiml_exp_required enum
CREATE TYPE aiml_exp_required AS ENUM (
    'Required',
    'Preferred',
    'Not Required'
);

-- Create coding_proficiency enum
CREATE TYPE coding_proficiency AS ENUM (
    'Basic',
    'Intermediate',
    'Advanced',
    'Expert'
);

-- Create education_required enum
CREATE TYPE education_required AS ENUM (
    'Top 30 CS program',
    'Ivy League',
    'Respected schools',
    'No requirement',
    'No bootcamps'
);

-- Create education_advanced_degree enum
CREATE TYPE education_advanced_degree AS ENUM (
    'PhD preferred',
    'Master''s preferred',
    'Not required'
);

-- Create autonomy_level enum
CREATE TYPE autonomy_level AS ENUM (
    'Low',
    'Medium',
    'High',
    'Very High'
);

-- Update jobs_dev table to use the new enum types
ALTER TABLE jobs_dev
    ALTER COLUMN company_stage TYPE company_stage USING company_stage::company_stage,
    ALTER COLUMN company_industry_vertical TYPE company_industry_vertical USING company_industry_vertical::company_industry_vertical,
    ALTER COLUMN role_category TYPE role_category USING role_category::role_category,
    ALTER COLUMN seniority TYPE seniority USING seniority::seniority,
    ALTER COLUMN work_arrangement TYPE work_arrangement[] USING work_arrangement::work_arrangement[],
    ALTER COLUMN role_status TYPE role_status USING role_status::role_status,
    ALTER COLUMN tech_breadth TYPE tech_breadth USING tech_breadth::tech_breadth,
    ALTER COLUMN tech_breadth_requirement TYPE tech_breadth USING tech_breadth_requirement::tech_breadth,
    ALTER COLUMN ai_ml_exp_required TYPE aiml_exp_required USING ai_ml_exp_required::aiml_exp_required,
    ALTER COLUMN coding_proficiency TYPE coding_proficiency USING coding_proficiency::coding_proficiency,
    ALTER COLUMN education_required TYPE education_required USING education_required::education_required,
    ALTER COLUMN education_advanced_degree TYPE education_advanced_degree USING education_advanced_degree::education_advanced_degree,
    ALTER COLUMN independent_work_capacity TYPE autonomy_level USING independent_work_capacity::autonomy_level,
    ALTER COLUMN independent_work TYPE autonomy_level USING independent_work::autonomy_level,
    ALTER COLUMN autonomy_level_required TYPE autonomy_level USING autonomy_level_required::autonomy_level; 