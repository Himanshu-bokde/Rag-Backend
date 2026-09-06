from redis import Redis

from app.core.config import settings


redis_connection = Redis.from_url(
    settings.REDIS_URL,
    decode_responses=True
)