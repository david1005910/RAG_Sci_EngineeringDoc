"""CSRF Protection Middleware using Double Submit Cookie Pattern"""

import secrets
import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# CSRF token settings
CSRF_COOKIE_NAME = "csrf_token"
CSRF_HEADER_NAME = "X-CSRF-Token"
CSRF_TOKEN_LENGTH = 32

# Methods that require CSRF protection
CSRF_PROTECTED_METHODS = {"POST", "PUT", "DELETE", "PATCH"}

# Paths exempt from CSRF protection (e.g., login, public endpoints)
CSRF_EXEMPT_PATHS = {
    "/api/v1/auth/login",
    "/api/v1/auth/register",
    "/api/v1/csrf-token",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/health",
    "/",
}


def generate_csrf_token() -> str:
    """Generate a cryptographically secure CSRF token"""
    return secrets.token_hex(CSRF_TOKEN_LENGTH)


class CSRFMiddleware(BaseHTTPMiddleware):
    """
    CSRF Protection Middleware using Double Submit Cookie pattern.

    - Sets a CSRF token in a cookie (accessible to JavaScript)
    - Validates the token from X-CSRF-Token header matches the cookie
    - Only validates for state-changing methods (POST, PUT, DELETE, PATCH)
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get or generate CSRF token
        csrf_token = request.cookies.get(CSRF_COOKIE_NAME)

        if not csrf_token:
            csrf_token = generate_csrf_token()

        # Check if CSRF validation is needed
        if request.method in CSRF_PROTECTED_METHODS:
            # Check if path is exempt
            path = request.url.path
            is_exempt = any(path.startswith(exempt) for exempt in CSRF_EXEMPT_PATHS)

            if not is_exempt:
                # Get token from header
                header_token = request.headers.get(CSRF_HEADER_NAME)
                cookie_token = request.cookies.get(CSRF_COOKIE_NAME)

                # Validate token
                if not header_token or not cookie_token:
                    logger.warning(f"CSRF token missing for {request.method} {path}")
                    return JSONResponse(
                        status_code=403,
                        content={"detail": "CSRF token missing"}
                    )

                if not secrets.compare_digest(header_token, cookie_token):
                    logger.warning(f"CSRF token mismatch for {request.method} {path}")
                    return JSONResponse(
                        status_code=403,
                        content={"detail": "CSRF token invalid"}
                    )

        # Process request
        response = await call_next(request)

        # Set CSRF cookie if not present or refresh it
        if not request.cookies.get(CSRF_COOKIE_NAME):
            response.set_cookie(
                key=CSRF_COOKIE_NAME,
                value=csrf_token,
                httponly=False,  # Must be accessible to JavaScript
                secure=False,  # Set to True in production with HTTPS
                samesite="lax",
                max_age=3600 * 24,  # 24 hours
                path="/",
            )

        return response
