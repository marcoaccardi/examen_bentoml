"""Endpoint tests for the distributed admission service.

Run against the running architecture (see README.md):
    docker compose up -d
    pytest -v tests/test_endpoints.py
"""

import time

import jwt
import pytest
import requests

BASE_URL = "http://localhost:3000"
TIMEOUT = 15

# Same secret as in src/auth/jwt_auth.py.
JWT_SECRET_KEY = "admission_secret_key"
JWT_ALGORITHM = "HS256"

STUDENT = {
    "gre_score": 309,
    "toefl_score": 108,
    "university_rating": 4,
    "sop": 3.0,
    "lor": 4.0,
    "cgpa": 7.94,
    "research": 0,
}


@pytest.fixture(scope="module")
def base_url():
    return BASE_URL


@pytest.fixture(scope="module")
def auth_token(base_url):
    """Fixture to obtain an authentication token."""
    credentials = {"username": "user123", "password": "password123"}
    response = requests.post(f"{base_url}/login", json=credentials, timeout=TIMEOUT)
    assert response.status_code == 200
    assert "token" in response.json()
    return response.json().get("token")


@pytest.fixture(scope="module")
def headers(auth_token):
    """Fixture for authenticated headers."""
    return {"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"}


def test_login_success(base_url, auth_token):
    """The token decodes with the service secret and identifies the user."""
    payload = jwt.decode(auth_token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    assert payload["sub"] == "user123"
    assert "exp" in payload


def test_login_failure(base_url):
    """Login with invalid credentials."""
    credentials = {"username": "wrong", "password": "wrong"}
    response = requests.post(f"{base_url}/login", json=credentials, timeout=TIMEOUT)

    assert response.status_code == 401
    response_data = response.json()
    assert isinstance(response_data, dict)
    assert response_data["detail"] == "Invalid credentials"


def test_predict_requires_token(base_url):
    """Prediction without a token is rejected."""
    response = requests.post(f"{base_url}/predict", json=STUDENT, timeout=TIMEOUT)
    assert response.status_code == 401


def test_single_prediction(base_url, headers):
    """Single prediction through the API service."""
    response = requests.post(f"{base_url}/predict", headers=headers, json=STUDENT, timeout=TIMEOUT)

    assert response.status_code == 200
    prediction = response.json()
    assert "prediction" in prediction
    assert isinstance(prediction["prediction"], list)
    assert len(prediction["prediction"]) == 1
    assert isinstance(prediction["prediction"][0], float)


def test_single_prediction_invalid_input(base_url, headers):
    """Out-of-range input is rejected by validation."""
    response = requests.post(
        f"{base_url}/predict", headers=headers, json={**STUDENT, "cgpa": 42}, timeout=TIMEOUT
    )

    assert response.status_code in (400, 422)
    assert "prediction" not in response.json()


def test_batch_prediction(base_url, headers):
    """Batch prediction: submit a job, poll its status until completion."""
    input_data = {"predictions": [STUDENT] * 3}

    # Submit the batch job.
    response = requests.post(
        f"{base_url}/batch_predict", headers=headers, json=input_data, timeout=TIMEOUT
    )

    assert response.status_code == 200
    job_data = response.json()
    assert "job_id" in job_data
    assert job_data["status"] == "pending"

    # Poll the status until completion.
    job_id = job_data["job_id"]
    max_retries = 10
    retry_count = 0
    status_data = None

    while retry_count < max_retries:
        status_response = requests.post(
            f"{base_url}/batch_status", headers=headers, json={"job_id": job_id}, timeout=TIMEOUT
        )

        assert status_response.status_code == 200
        status_data = status_response.json()

        if status_data["status"] == "completed":
            break
        elif status_data["status"] == "failed":
            pytest.fail(f"Batch job failed: {status_data.get('error')}")

        time.sleep(1)
        retry_count += 1

    # Verify the results after the loop.
    assert status_data is not None
    assert status_data["status"] == "completed"
    assert isinstance(status_data["predictions"], list)
    assert len(status_data["predictions"]) == 3


def test_invalid_batch_status(base_url, headers):
    """Asking for an unknown job id returns a 404."""
    response = requests.post(
        f"{base_url}/batch_status", headers=headers, json={"job_id": "invalid_id"}, timeout=TIMEOUT
    )

    assert response.status_code == 404
    response_data = response.json()
    assert isinstance(response_data, dict)
    assert response_data["detail"] == "Job invalid_id not found"
