import httpx
import logging
from typing import Dict, Any
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class WebhookProxy:
    def __init__(self, local_webhook_url: str = "http://localhost:8000/webhook/retell"):
        self.local_webhook_url = local_webhook_url
        self.client = httpx.AsyncClient()

    async def forward_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward a webhook from production to localhost.
        
        Args:
            payload: The webhook payload from Retell
            
        Returns:
            Dict containing the response from localhost
        """
        try:
            logger.info(f"Forwarding webhook to localhost: {self.local_webhook_url}")
            
            # Forward the webhook to localhost
            response = await self.client.post(
                self.local_webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                logger.error(f"Local webhook returned error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Local webhook error: {response.text}"
                )
            
            logger.info("Successfully forwarded webhook to localhost")
            return response.json()
            
        except Exception as e:
            logger.error(f"Error forwarding webhook: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error forwarding webhook: {str(e)}"
            ) 