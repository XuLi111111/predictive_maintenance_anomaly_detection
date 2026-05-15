from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import live, models, predict, report, stream, upload
from app.core.config import settings

app = FastAPI(title="pump.detect API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(models.router, prefix="/api", tags=["models"])
app.include_router(predict.router, prefix="/api", tags=["predict"])
app.include_router(report.router, prefix="/api", tags=["report"])
app.include_router(stream.router, prefix="/api", tags=["stream"])
app.include_router(live.router, prefix="/api", tags=["live"])


@app.get("/api/health")
def health():
    return {"status": "ok", "version": app.version}
