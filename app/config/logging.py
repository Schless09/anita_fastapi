import logging
import sys
from typing import Dict, Any
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add extra fields if they exist
        if hasattr(record, 'extra'):
            log_data.update(record.extra)

        # Add exception info if it exists
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data)

def setup_logging(level: str = 'INFO') -> None:
    """
    Set up logging configuration for the application.
    """
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(console_handler)

    # Create file handler
    file_handler = logging.FileHandler('app.log')
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)

    # Set logging level for specific modules
    logging.getLogger('httpx').setLevel('WARNING')
    logging.getLogger('openai').setLevel('WARNING')
    logging.getLogger('pinecone').setLevel('WARNING')
    logging.getLogger('supabase').setLevel('WARNING')

    logging.info("Logging system initialized") 