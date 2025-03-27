import time
from functools import wraps
from fastapi import HTTPException
import redis
import os

# Initialize Redis client
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=int(os.getenv('REDIS_DB', 0))
)

def RateLimiter(times: int, seconds: int):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get the request object from args
            request = next((arg for arg in args if hasattr(arg, 'client')), None)
            if not request:
                raise HTTPException(status_code=400, detail="Request object not found")
            
            # Create a unique key for this endpoint and client
            client_ip = request.client.host
            key = f"rate_limit:{func.__name__}:{client_ip}"
            
            # Get current count
            current = redis_client.get(key)
            if current is None:
                # First request, set initial count and expiry
                redis_client.setex(key, seconds, 1)
                return await func(*args, **kwargs)
            
            current_count = int(current)
            if current_count >= times:
                raise HTTPException(
                    status_code=429,
                    detail=f"Too many requests. Please try again in {seconds} seconds."
                )
            
            # Increment counter
            redis_client.incr(key)
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator 