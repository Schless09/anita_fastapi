"""
Configuration constants for the application.
These values can be overridden by environment variables.
"""
from typing import Final, Dict

# Call Processing
MIN_CALL_DURATION_SECONDS: Final[int] = 30  # Minimum duration for a call to be considered valid
MAX_CALL_DURATION_SECONDS: Final[int] = 1800  # Maximum duration (30 min) for a call

# Matching
MATCH_THRESHOLD: Final[float] = 0.3  # Minimum similarity score for a job match (0-1)
MAX_MATCHES_PER_CANDIDATE: Final[int] = 10  # Maximum number of job matches to return per candidate

# Matching Weights (must sum to 1.0)
MATCH_WEIGHTS: Final[Dict[str, float]] = {
    'skills_and_tech': 0.35,      # Technical skills, languages, frameworks
    'experience': 0.25,           # Years of experience, past roles
    'role_alignment': 0.20,       # Job title, role category, seniority
    'preferences': 0.15,          # Location, company size, industry
    'culture': 0.05               # Company culture, work style
}

# Skills Matching Weights (must sum to 1.0)
SKILLS_MATCH_WEIGHTS: Final[Dict[str, float]] = {
    'must_have_skills': 0.6,      # Required/must-have skills
    'nice_to_have_skills': 0.2,   # Preferred/nice-to-have skills
    'tech_stack': 0.2             # Specific technologies
}

# Experience Matching Weights (must sum to 1.0)
EXPERIENCE_MATCH_WEIGHTS: Final[Dict[str, float]] = {
    'years_of_experience': 0.4,    # Total years of experience
    'domain_expertise': 0.3,       # Relevant industry/domain experience
    'role_specific_exp': 0.3      # Experience in similar roles
}

# Vector Search
VECTOR_DIMENSION: Final[int] = 1536  # OpenAI embedding dimension
TOP_K_SIMILAR_JOBS: Final[int] = 20  # Number of similar jobs to return in vector search

# Email
MAX_RETRY_ATTEMPTS: Final[int] = 3  # Maximum number of retry attempts for email sending
EMAIL_RETRY_DELAY_SECONDS: Final[int] = 5  # Delay between retry attempts

# Webhook Processing
WEBHOOK_TIMEOUT_SECONDS: Final[int] = 30  # Timeout for webhook processing
MAX_WEBHOOK_RETRIES: Final[int] = 3  # Maximum number of webhook retry attempts

# Rate Limiting
MAX_REQUESTS_PER_MINUTE: Final[int] = 60  # Maximum number of API requests per minute
RATE_LIMIT_WINDOW_SECONDS: Final[int] = 60  # Time window for rate limiting

# Cache
CACHE_TTL_SECONDS: Final[int] = 3600  # Time-to-live for cached items (1 hour)
MAX_CACHE_SIZE: Final[int] = 1000  # Maximum number of items in cache

# Database
DB_CONNECTION_TIMEOUT: Final[int] = 30  # Database connection timeout in seconds
MAX_DB_RETRIES: Final[int] = 3  # Maximum number of database retry attempts

# API
API_VERSION: Final[str] = "v1"  # API version
DEFAULT_PAGE_SIZE: Final[int] = 50  # Default number of items per page
MAX_PAGE_SIZE: Final[int] = 100  # Maximum number of items per page 