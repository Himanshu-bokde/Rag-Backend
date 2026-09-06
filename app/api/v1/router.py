from fastapi import APIRouter

from  app.api.v1.documents import route as documents_router
from  app.api.v1.chat import route as chat_router

route = APIRouter(
    prefix="/api/v1"
)


route.include_router(documents_router)
route.include_router(chat_router)