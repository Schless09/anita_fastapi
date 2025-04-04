from typing import Dict, Any, Optional
import logging
import httpx
from app.config import get_settings

# Set up logging
logger = logging.getLogger(__name__)

settings = get_settings()

class RetellService:
    def __init__(self):
        self.api_key = settings.retell_api_key
        self.phone_number = settings.retell_from_number
        self.webhook_url = settings.retell_webhook_url
        self.agent_id = settings.retell_agent_id
        self.api_base = settings.retell_api_base
        logger.info(f"Initialized RetellService with phone number: {self.phone_number}")

    async def schedule_call(
        self,
        candidate_id: str,
        dynamic_variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Schedule a call with a candidate using Retell.
        """
        try:
            logger.info(f"Scheduling call for candidate: {candidate_id}")
            
            # Format phone number to E.164 format
            phone = dynamic_variables.get('phone', '')
            if not phone:
                raise ValueError("Phone number is required")
            
            # Clean the phone number
            phone = ''.join(filter(str.isdigit, phone))
            
            # Handle different formats
            if phone.startswith('1') and len(phone) == 11:  # US number with country code
                phone = f"+{phone}"
            elif len(phone) == 10:  # US number without country code
                phone = f"+1{phone}"
            elif not phone.startswith('+') and len(phone) > 10:  # International number
                phone = f"+{phone}"
            
            # Validate final format
            if not phone.startswith('+') or len(phone) < 10:
                raise ValueError(f"Invalid phone number format: {phone}. Must be in E.164 format (e.g., +12137774445)")
            
            logger.info(f"Formatted phone number: {phone}")
            
            # Convert all dynamic variables to strings
            string_vars = {
                key: str(value) if value is not None else ""
                for key, value in dynamic_variables.items()
            }
            
            # Get candidate data to get phone number
            async with httpx.AsyncClient() as client:
                # Create the call using Retell API
                response = await client.post(
                    f"{self.api_base}/create-phone-call",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "from_number": self.phone_number,
                        "to_number": phone,  # Use formatted phone number
                        "agent_id": self.agent_id,
                        "metadata": {
                            "candidate_id": candidate_id
                        },
                        "retell_llm_dynamic_variables": string_vars  # Use string-converted variables
                    }
                )
                
                if response.status_code != 201:
                    error_msg = f"Retell API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                
                call_data = response.json()
                logger.info(f"Successfully scheduled call: {call_data}")
                return call_data
            
        except Exception as e:
            logger.error(f"Error scheduling call: {str(e)}")
            raise Exception(f"Error scheduling call with Retell: {str(e)}")

    async def get_call_status(self, call_id: str) -> Dict[str, Any]:
        """
        Get the status of a scheduled call.
        """
        try:
            logger.info(f"Getting status for call: {call_id}")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_base}/get-call/{call_id}",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                )
                
                if response.status_code != 200:
                    error_msg = f"Retell API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                
                call_data = response.json()
                logger.info(f"Successfully got call status: {call_data}")
                return call_data
            
        except Exception as e:
            logger.error(f"Error getting call status: {str(e)}")
            raise Exception(f"Error getting call status from Retell: {str(e)}")

    async def get_call(self, call_id: str) -> Dict[str, Any]:
        """
        Get call data from Retell API.
        """
        try:
            logger.info(f"Getting data for call: {call_id}")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_base}/get-call/{call_id}",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                )
                
                if response.status_code != 200:
                    error_msg = f"Retell API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                
                call_data = response.json()
                logger.info(f"Successfully retrieved call data: {call_data}")
                return call_data
            
        except Exception as e:
            logger.error(f"Error getting call data: {str(e)}")
            raise Exception(f"Error getting call data from Retell: {str(e)}")

    async def cancel_call(self, call_id: str) -> Dict[str, Any]:
        """
        Cancel a scheduled call.
        """
        try:
            logger.info(f"Canceling call: {call_id}")
            
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{self.api_base}/v2/call/{call_id}",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                )
                
                if response.status_code != 200:
                    error_msg = f"Retell API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                
                call_data = response.json()
                logger.info(f"Successfully canceled call: {call_data}")
                return call_data
            
        except Exception as e:
            logger.error(f"Error canceling call: {str(e)}")
            raise Exception(f"Error canceling call with Retell: {str(e)}") 