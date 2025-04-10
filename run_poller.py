import asyncio
import os
import sys
from loguru import logger

# Add project root to path to allow imports from app and anita
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now import necessary components
from app.config.settings import get_settings
from app.config.supabase import get_supabase_client
from anita.services.email_service import EmailService
from anita.services.slack_service import SlackService
from anita.services.inbound_email_service import InboundEmailService
from anita.services.gmail_polling_service import poll_gmail_for_updates

async def main():
    logger.info("Initializing services for background worker...")
    settings = get_settings()
    supabase_client = get_supabase_client()
    
    # Instantiate services needed for polling
    email_service = EmailService(settings=settings)
    slack_service = SlackService(settings=settings)
    inbound_service = InboundEmailService(
        settings=settings,
        supabase_client=supabase_client,
        email_service=email_service,
        slack_service=slack_service
    )
    
    logger.info("Starting Gmail polling loop...")
    await poll_gmail_for_updates(
        settings=settings,
        email_service=email_service,
        inbound_service=inbound_service,
        poll_interval_seconds=settings.gmail_poll_interval_seconds # Use configured interval
    )

if __name__ == "__main__":
    # Configure Loguru sink for console output
    logger.add(sys.stderr, level="DEBUG") # Adjust level as needed
    logger.info("Starting run_poller.py script...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Polling worker stopped by user.")
    except Exception as e:
        logger.exception(f"Polling worker crashed: {e}")
        sys.exit(1) # Exit with error code if crashes 