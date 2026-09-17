"""JWT authentication for the admission API.

All the auth logic lives here so the service file stays focused on the endpoints.
"""

import os
from datetime import datetime, timedelta, timezone

import jwt
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# Read from the environment when set (docker-compose does this), with a default
# so `bentoml serve` and pytest also work without any setup.
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "admission_secret_key")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 1

# Simple hardcoded users.
USERS = {
    "user123": "password123",
    "admin": "admin123",
}

# The routes that need a token. /login stays open - it is where tokens come from.
PROTECTED_PATHS = ("/predict", "/batch_predict", "/batch_status")


def create_jwt_token(user_id: str) -> str:
    """Create a signed token that expires after one hour."""
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def verify_credentials(username: str, password: str) -> bool:
    return USERS.get(username) == password


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Check the JWT before the request reaches the protected endpoints."""

    async def dispatch(self, request, call_next):
        if not request.url.path.endswith(PROTECTED_PATHS):
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return JSONResponse(
                status_code=401, content={"detail": "Missing authentication token"}
            )

        # Expected format: "Bearer <token>".
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return JSONResponse(
                status_code=401, content={"detail": "Invalid authorization header"}
            )

        try:
            payload = jwt.decode(parts[1], JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        except jwt.ExpiredSignatureError:
            return JSONResponse(status_code=401, content={"detail": "Token has expired"})
        except jwt.InvalidTokenError:
            return JSONResponse(status_code=401, content={"detail": "Invalid token"})

        request.state.user = payload.get("sub")
        return await call_next(request)
