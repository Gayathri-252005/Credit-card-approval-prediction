"""
main.py — FastAPI application entry point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import predict, batch, health, eligibility

app = FastAPI(
    title="Credit Risk & Eligibility Decisioning API",
    description=(
        "AI-powered credit card application screening platform. "
        "Supports analyst single-screening, compliance batch review, "
        "and customer self-service pre-qualification."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5500",   # VS Code Live Server
        "http://localhost:5500",   # VS Code Live Server (localhost)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router,      prefix="/api/v1", tags=["Health"])
app.include_router(predict.router,     prefix="/api/v1", tags=["Analyst"])
app.include_router(batch.router,       prefix="/api/v1", tags=["Compliance"])
app.include_router(eligibility.router, prefix="/api/v1", tags=["Customer"])


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Credit Risk & Eligibility Decisioning API",
        "version": "1.0.0",
        "docs": "/docs",
    }
