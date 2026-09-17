"""Tests for the admission prediction service.

Three groups:
  1. Login API   - valid credentials -> token, invalid credentials -> 401
  2. JWT auth    - missing / invalid / expired / valid token on /predict
  3. Prediction  - valid input -> prediction, invalid input -> error

Run against a running service (see README.md):
    pytest -v
"""

import jwt
import pytest
import requests

from conftest import JWT_ALGORITHM, JWT_SECRET_KEY, INVALID_USER, VALID_INPUT, VALID_USER, make_token

TIMEOUT = 10


# ---------------------------------------------------------------------------
# 1. Login API
# ---------------------------------------------------------------------------
def test_login_success(base_url):
    """Correct credentials return a valid JWT token."""
    response = requests.post(
        f"{base_url}/login", json={"credentials": VALID_USER}, timeout=TIMEOUT
    )

    assert response.status_code == 200
    body = response.json()
    assert "token" in body

    # The token decodes with the service secret and identifies the user.
    payload = jwt.decode(body["token"], JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    assert payload["sub"] == VALID_USER["username"]
    assert "exp" in payload


def test_login_failure(base_url):
    """Wrong credentials return a 401."""
    response = requests.post(
        f"{base_url}/login", json={"credentials": INVALID_USER}, timeout=TIMEOUT
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials"


# ---------------------------------------------------------------------------
# 2. JWT authentication on /predict
# ---------------------------------------------------------------------------
def test_predict_missing_token(base_url):
    """No Authorization header -> 401."""
    response = requests.post(
        f"{base_url}/predict", json={"input_data": VALID_INPUT}, timeout=TIMEOUT
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Missing authentication token"


def test_predict_invalid_token(base_url):
    """A garbage token -> 401."""
    response = requests.post(
        f"{base_url}/predict",
        headers={"Authorization": "Bearer not.a.real.token"},
        json={"input_data": VALID_INPUT},
        timeout=TIMEOUT,
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid token"


def test_predict_expired_token(base_url):
    """A token signed with the right secret but already expired -> 401."""
    expired = make_token(expires_in_hours=-1)

    response = requests.post(
        f"{base_url}/predict",
        headers={"Authorization": f"Bearer {expired}"},
        json={"input_data": VALID_INPUT},
        timeout=TIMEOUT,
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Token has expired"


def test_predict_valid_token(base_url, headers):
    """A valid token passes authentication."""
    response = requests.post(
        f"{base_url}/predict", headers=headers, json={"input_data": VALID_INPUT}, timeout=TIMEOUT
    )

    assert response.status_code == 200


# ---------------------------------------------------------------------------
# 3. Prediction API
# ---------------------------------------------------------------------------
def test_predict_valid_input(base_url, headers):
    """A valid input returns a prediction in the expected format."""
    response = requests.post(
        f"{base_url}/predict", headers=headers, json={"input_data": VALID_INPUT}, timeout=TIMEOUT
    )

    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body

    prediction = body["prediction"]
    assert isinstance(prediction, list)
    assert len(prediction) == 1
    assert isinstance(prediction[0], float)
    # The chance of admission is a probability.
    assert 0.0 <= prediction[0] <= 1.0


@pytest.mark.parametrize(
    "bad_input",
    [
        # missing feature: cgpa removed
        {k: v for k, v in VALID_INPUT.items() if k != "cgpa"},
        # out of range: GRE score max is 340
        {**VALID_INPUT, "gre_score": 9999},
        # wrong type
        {**VALID_INPUT, "gre_score": "three hundred"},
    ],
)
def test_predict_invalid_input(base_url, headers, bad_input):
    """Invalid input data returns an error, not a prediction."""
    response = requests.post(
        f"{base_url}/predict", headers=headers, json={"input_data": bad_input}, timeout=TIMEOUT
    )

    assert response.status_code in (400, 422)
    assert "prediction" not in response.json()
