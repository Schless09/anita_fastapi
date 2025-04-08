from fastapi import APIRouter, Request, Depends
from app.services.webhook_proxy import WebhookProxy
from app.dependencies import get_webhook_proxy

router = APIRouter()

@router.post("/retell/proxy")
async def proxy_webhook(
    request: Request,
    webhook_proxy: WebhookProxy = Depends(get_webhook_proxy)
):
    """
    Proxy endpoint that receives webhooks from Retell and forwards them to localhost.
    """
    # Get the webhook payload
    payload = await request.json()
    
    # Forward to localhost
    return await webhook_proxy.forward_webhook(payload) 