from functools import lru_cache
from fastapi import Depends
from supabase._async.client import AsyncClient

from app.config.settings import Settings, get_settings
from app.config.utils import get_table_name
from app.config.supabase import get_supabase_client
from app.services.openai_service import OpenAIService
from app.services.vector_service import VectorService
from app.services.job_service import JobService
from app.services.matching_service import MatchingService
from app.services.candidate_service import CandidateService
from app.services.retell_service import RetellService
from app.agents.brain_agent import BrainAgent

# Cached settings
@lru_cache()
def get_cached_settings() -> Settings:
    # This should be the single source of truth for settings
    return get_settings()

# Provider for OpenAI Service
def get_openai_service(settings: Settings = Depends(get_cached_settings)) -> OpenAIService:
    return OpenAIService(settings=settings)

# Provider for Supabase Client
def get_supabase_client_dependency() -> AsyncClient:
    # Assuming get_supabase_client handles pooling/caching if necessary
    return get_supabase_client()

# Provider for Retell Service (Updated)
def get_retell_service(
    settings: Settings = Depends(get_cached_settings)
) -> RetellService:
    return RetellService(settings=settings)

# Provider for Vector Service
def get_vector_service(
    settings: Settings = Depends(get_cached_settings),
    openai_service: OpenAIService = Depends(get_openai_service),
    supabase_client: AsyncClient = Depends(get_supabase_client_dependency)
) -> VectorService:
    candidates_table = get_table_name("candidates")
    jobs_table = get_table_name("jobs")
    return VectorService(
        openai_service=openai_service,
        supabase_client=supabase_client,
        candidates_table=candidates_table,
        jobs_table=jobs_table
    )

# Provider for Job Service
def get_job_service(
    supabase_client: AsyncClient = Depends(get_supabase_client_dependency),
    vector_service: VectorService = Depends(get_vector_service),
    openai_service: OpenAIService = Depends(get_openai_service)
) -> JobService:
    return JobService(
        supabase_client=supabase_client,
        vector_service=vector_service,
        openai_service=openai_service
    )

# Provider for Matching Service
def get_matching_service(
    openai_service: OpenAIService = Depends(get_openai_service),
    vector_service: VectorService = Depends(get_vector_service),
    supabase_client: AsyncClient = Depends(get_supabase_client_dependency),
    settings: Settings = Depends(get_cached_settings)
) -> MatchingService:
    return MatchingService(
        openai_service=openai_service,
        vector_service=vector_service,
        supabase_client=supabase_client,
        settings=settings
    )

# Provider for Candidate Service (Updated)
def get_candidate_service(
    supabase_client: AsyncClient = Depends(get_supabase_client_dependency),
    retell_service: RetellService = Depends(get_retell_service),
    openai_service: OpenAIService = Depends(get_openai_service),
    settings: Settings = Depends(get_cached_settings)
) -> CandidateService:
    return CandidateService(
        supabase_client=supabase_client,
        retell_service=retell_service,
        openai_service=openai_service,
        settings=settings
    )

# Provider for Brain Agent
def get_brain_agent(
    supabase_client: AsyncClient = Depends(get_supabase_client_dependency),
    candidate_service: CandidateService = Depends(get_candidate_service),
    openai_service: OpenAIService = Depends(get_openai_service),
    matching_service: MatchingService = Depends(get_matching_service),
    retell_service: RetellService = Depends(get_retell_service),
    vector_service: VectorService = Depends(get_vector_service),
    settings: Settings = Depends(get_cached_settings)
) -> BrainAgent:
    return BrainAgent(
        supabase_client=supabase_client,
        candidate_service=candidate_service,
        openai_service=openai_service,
        matching_service=matching_service,
        retell_service=retell_service,
        vector_service=vector_service,
        settings=settings
    ) 