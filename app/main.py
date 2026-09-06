from fastapi import FastAPI

from app.api.v1.router import route

app = FastAPI(
    title="RAG API",
    version="1.0.0"
)


app.include_router(route)

@app.get('/health')
def health():
    return {"status":"ok"}