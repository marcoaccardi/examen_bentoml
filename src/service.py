"""BentoML prediction service with JWT authentication.

Usage (from the project root):
    bentoml serve src.service:AdmissionPredictionService --port 3000

Endpoints:
    POST /login    {"credentials": {"username": ..., "password": ...}} -> {"token": ...}
    POST /predict  {"input_data": {... 7 features ...}}                -> {"prediction": [float]}
                   needs the header  Authorization: Bearer <token>
"""

from datetime import datetime, timedelta, timezone

import bentoml
import pandas as pd
import jwt
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# Secret used to sign the tokens.
JWT_SECRET_KEY = "admission_secret_key"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 1

# Simple hardcoded users
USERS = {
    "user123": "password123",
    "admin": "admin123",
}

# The feature order used at training time - predictions are positional.
FEATURE_ORDER = [
    "gre_score",
    "toefl_score",
    "university_rating",
    "sop",
    "lor",
    "cgpa",
    "research",
]


class Credentials(BaseModel):
    username: str
    password: str


class AdmissionInput(BaseModel):
    """The 7 input features, with the valid ranges from the dataset.

    The bounds make pydantic reject impossible values (e.g. a GRE score of 9999)
    with a 400 error instead of sending them to the model.
    """

    gre_score: int = Field(..., ge=0, le=340)
    toefl_score: int = Field(..., ge=0, le=120)
    university_rating: int = Field(..., ge=1, le=5)
    sop: float = Field(..., ge=0, le=5)
    lor: float = Field(..., ge=0, le=5)
    cgpa: float = Field(..., ge=0, le=10)
    research: int = Field(..., ge=0, le=1)


def create_jwt_token(user_id: str) -> str:
    """Create a signed token that expires after one hour."""
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Check the JWT on /predict before the request reaches the model.

    /login stays open - it is where clients get their token.
    """

    async def dispatch(self, request, call_next):
        if not request.url.path.endswith("/predict"):
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return JSONResponse(
                status_code=401, content={"detail": "Missing authentication token"}
            )

        # Expected format: "Bearer <token>". Check it to avoid an IndexError below.
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


# name= sets the bento name used by bentoml build and containerize.
@bentoml.service(name="admission_prediction_service")
class AdmissionPredictionService:
    def __init__(self) -> None:
        # Load the model once at startup, not on every request.
        self.model = bentoml.sklearn.load_model("admission_lr:latest")

    @bentoml.api(route="/login")
    def login(self, credentials: Credentials):
        if USERS.get(credentials.username) == credentials.password:
            return {"token": create_jwt_token(credentials.username)}
        return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})

    @bentoml.api(route="/predict")
    def predict(self, input_data: AdmissionInput) -> dict:
        # One-row DataFrame with the training column order, so scikit-learn
        # gets the same feature names it was fitted with.
        features = pd.DataFrame(
            [[getattr(input_data, name) for name in FEATURE_ORDER]],
            columns=FEATURE_ORDER,
        )
        # clip: a linear model can go slightly outside [0, 1] for extreme inputs.
        prediction = self.model.predict(features).clip(0.0, 1.0)
        return {"prediction": prediction.tolist()}


AdmissionPredictionService.add_asgi_middleware(JWTAuthMiddleware)
