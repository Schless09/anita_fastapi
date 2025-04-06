from .settings import get_settings

def get_table_name(base_name: str) -> str:
    """Returns the full table name with environment suffix (_dev or _prod)."""
    settings = get_settings()
    suffix = "_prod" if settings.environment == "production" else "_dev"
    return f"{base_name}{suffix}" 