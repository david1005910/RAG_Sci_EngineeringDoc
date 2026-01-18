"""Bio-RAG API Application Entry Point"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.core.csrf import CSRFMiddleware
from src.core.database import init_db, close_db
from src.api.v1 import auth, search, chat, library, trends, vectordb

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} API...")
    logger.info(f"Environment: {settings.APP_ENV}")

    # Initialize database (optional for development)
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning(f"Database initialization failed: {e}")
        logger.warning("Running in limited mode without database")

    yield

    # Shutdown
    logger.info("Shutting down...")
    try:
        await close_db()
        logger.info("Database connections closed")
    except Exception:
        pass


# Create FastAPI application
app = FastAPI(
    title="Bio-RAG API",
    description="AI-powered biomedical research platform with RAG capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "X-CSRF-Token"],
    expose_headers=["X-CSRF-Token"],
)

# Add CSRF protection middleware
app.add_middleware(CSRFMiddleware)

# Register API routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(search.router, prefix="/api/v1", tags=["Search"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(library.router, prefix="/api/v1/library", tags=["Library"])
app.include_router(trends.router, prefix="/api/v1/trends", tags=["Trends"])
app.include_router(vectordb.router, prefix="/api/v1/vectordb", tags=["VectorDB"])


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "environment": settings.APP_ENV
    }


@app.get("/api/v1/csrf-token", tags=["Security"])
async def get_csrf_token(request: Request):
    """Get CSRF token - this endpoint sets the CSRF cookie"""
    from src.core.csrf import CSRF_COOKIE_NAME
    token = request.cookies.get(CSRF_COOKIE_NAME, "")
    return {"csrf_token": token}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
