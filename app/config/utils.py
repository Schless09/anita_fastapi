from .settings import get_settings

def get_table_name(table_name: str, settings) -> str:
    """
    Get the appropriate table name based on the environment.
    
    Args:
        table_name: Base table name
        settings: Application settings containing environment info
        
    Returns:
        Table name with appropriate suffix:
        - For development and staging: {table_name}_dev
        - For production: {table_name}_prod
    """
    # Use _dev suffix for both development and staging environments
    suffix = "_prod" if settings.environment == "production" else "_dev"
    return f"{table_name}{suffix}" 