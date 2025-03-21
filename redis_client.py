from upstash_redis import Redis
import os
from typing import Optional, Dict, Any
import json
from datetime import datetime

class RedisClient:
    def __init__(self):
        redis_url = os.getenv('REDIS_URL')
        redis_token = os.getenv('REDIS_TOKEN')
        if not redis_url or not redis_token:
            raise ValueError("Redis URL and token must be set in environment variables")
        self.redis = Redis(url=redis_url, token=redis_token)
        
    async def set_job_status(self, job_id: str, status_data: Dict[str, Any]) -> bool:
        """Store job status in Redis."""
        try:
            # Convert datetime objects to ISO format strings
            serializable_data = self._prepare_for_json(status_data)
            # Store with 24-hour expiration (86400 seconds)
            return await self.redis.set(
                f"job_status:{job_id}",
                json.dumps(serializable_data),
                ex=86400
            )
        except Exception as e:
            print(f"Error setting job status: {str(e)}")
            return False

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve job status from Redis."""
        try:
            data = await self.redis.get(f"job_status:{job_id}")
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            print(f"Error getting job status: {str(e)}")
            return None

    async def delete_job_status(self, job_id: str) -> bool:
        """Delete job status from Redis."""
        try:
            return await self.redis.delete(f"job_status:{job_id}")
        except Exception as e:
            print(f"Error deleting job status: {str(e)}")
            return False

    def _prepare_for_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert datetime objects to ISO format strings for JSON serialization."""
        processed_data = {}
        for key, value in data.items():
            if isinstance(value, datetime):
                processed_data[key] = value.isoformat()
            elif isinstance(value, dict):
                processed_data[key] = self._prepare_for_json(value)
            else:
                processed_data[key] = value
        return processed_data

# Initialize Redis client
redis_client = RedisClient() 