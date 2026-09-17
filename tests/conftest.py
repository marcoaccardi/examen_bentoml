"""Shared fixtures for the admission service tests.

The tests call the running service over HTTP (the container started with docker run),
so they do not import anything from src/.
"""

import os
from datetime import datetime, timedelta, timezone

import jwt
import pytest
import requests

# Default URL of the running service, can be overridden:
#   BASE_URL=http://localhost:3001 pytest -v
BASE_URL = os.environ.get("BASE_URL", "http://localhost:3000")

# Same secret as in src/service.py - needed to create an expired token in the tests.
JWT_SECRET_KEY = "admission_secret_key"
JWT_ALGORITHM = "HS256"

VALID_USER = {"username": "user123", "password": "password123"}
INVALID_USER = {"username": "wrong_user", "password": "wrong_password"}

# A valid example input used by the prediction tests.
VALID_INPUT = {
    "gre_score": 309,
    "toefl_score": 108,
    "university_rating": 4,
    "sop": 3.0,
    "lor": 4.0,
    "cgpa": 7.94,
    "research": 0,
}


@pytest.fixture(scope="module")
def base_url() -> str:
    return BASE_URL


@pytest.fixture(scope="module")
def auth_token(base_url: str) -> str:
    """Log in once for the whole test module and reuse the token."""
    response = requests.post(
        f"{base_url}/login", json={"credentials": VALID_USER}, timeout=10
    )
    assert response.status_code == 200, f"login failed: {response.text}"
    token = response.json().get("token")
    assert token, "login returned no token"
    return token


@pytest.fixture(scope="module")
def headers(auth_token: str) -> dict:
    return {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
    }


def make_token(user_id: str = "user123", expires_in_hours: float = 1.0) -> str:
    """Sign a token locally. With a negative expires_in_hours it is already expired."""
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(hours=expires_in_hours),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
