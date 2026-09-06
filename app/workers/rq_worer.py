from rq import Queue
from app.core.redis import redis_connection


decument_queue = Queue(
    "document-processing",
    connection=redis_connection,
    default_timeout=900
)