import asyncio
import logging
import sys # Import sys
import os # Import os
from dotenv import load_dotenv

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Configure logging similar to main.py
logger = logging.getLogger('scripts')
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s [%(filename)s:%(lineno)d] - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Suppress noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

async def run_test():
    """Initialize services and run the Slack connection test."""
    load_dotenv() # Load .env file
    
    logger.info("Attempting to initialize settings and SlackService...")
    
    try:
        # Import necessary components *after* load_dotenv and path setup
        from app.config.settings import get_settings
        from app.services.slack_service import SlackService
        
        settings = get_settings()
        slack_service = SlackService(settings=settings)
        
        logger.info("SlackService initialized. Running test connection...")
        
        # Run the synchronous test method
        # If SlackService methods become async, you'd await them here
        success = slack_service.test_connection()
        
        if success:
            logger.info("Slack test concluded successfully.")
        else:
            logger.error("Slack test concluded with errors.")
            
    except ImportError as ie:
        logger.error(f"ImportError: {ie}. Ensure your PYTHONPATH is correct or run from the project root.")
    except Exception as e:
        logger.error(f"An error occurred during initialization or testing: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(run_test()) 