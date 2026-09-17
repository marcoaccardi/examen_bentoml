"""Admission prediction with an API service and two workers.

    Client -> AdmissionAPI -> SinglePredictor  (low latency)
                           -> BatchPredictor   (async batch jobs)

The API service handles routing, validation and authentication; the workers do
the inference. Each one runs as its own process, so a long batch job never
blocks single predictions.

Run everything at once (BentoML starts the workers automatically):
    bentoml serve src.service_batch:AdmissionAPI --port 3000

Or run each service separately, like docker-compose does:
    bentoml serve src.service_batch:AdmissionAPI    --host 0.0.0.0 --port 3000
    bentoml serve src.service_batch:SinglePredictor --host 0.0.0.0 --port 3001
    bentoml serve src.service_batch:BatchPredictor  --host 0.0.0.0 --port 3002
"""

import asyncio
import os
import uuid

import bentoml
import pandas as pd
from pydantic import ValidationError
from starlette.responses import JSONResponse

from src.auth.jwt_auth import JWTAuthMiddleware, create_jwt_token, verify_credentials
from src.models.input_model import FEATURE_ORDER, AdmissionBatchInput, AdmissionInput

MODEL_TAG = "admission_lr:latest"

# When these are set (by docker-compose), the API calls the workers over the network.
# When not set, bentoml.depends starts the workers as local child processes.
SINGLE_PREDICTOR_URL = os.environ.get("SINGLE_PREDICTOR_URL")
BATCH_PREDICTOR_URL = os.environ.get("BATCH_PREDICTOR_URL")


def to_frame(items: list[AdmissionInput]) -> pd.DataFrame:
    """Build a DataFrame with the columns in training order."""
    return pd.DataFrame([item.as_row() for item in items], columns=FEATURE_ORDER)


@bentoml.service(name="single_predictor", workers=1)
class SinglePredictor:
    """Worker 1: one prediction per request, as fast as possible."""

    def __init__(self) -> None:
        self.model = bentoml.sklearn.load_model(MODEL_TAG)

    @bentoml.api
    def predict(self, input_data: AdmissionInput) -> dict:
        prediction = self.model.predict(to_frame([input_data])).clip(0.0, 1.0)
        return {"prediction": prediction.tolist()}


@bentoml.service(name="batch_predictor", workers=1)
class BatchPredictor:
    """Worker 2: accepts a batch, returns a job id right away, computes in the background.

    The client then polls the status until "completed" or "failed".
    workers=1 because the jobs dict lives in this process's memory - with several
    workers the status requests could land in a process that does not know the job.
    """

    def __init__(self) -> None:
        self.model = bentoml.sklearn.load_model(MODEL_TAG)
        # job_id -> job info (status, predictions, error)
        self.jobs: dict[str, dict] = {}

    async def _run_job(self, job_id: str, items: list[AdmissionInput]) -> None:
        try:
            predictions = self.model.predict(to_frame(items)).clip(0.0, 1.0)
            self.jobs[job_id] = {
                "job_id": job_id,
                "status": "completed",
                "predictions": [float(p) for p in predictions],
            }
        except Exception as exc:
            self.jobs[job_id] = {"job_id": job_id, "status": "failed", "error": str(exc)}

    @bentoml.api
    async def submit(self, batch: AdmissionBatchInput) -> dict:
        """Register the job and return immediately, without waiting for the result."""
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {"job_id": job_id, "status": "pending"}
        # create_task runs the job in the background.
        asyncio.create_task(self._run_job(job_id, batch.predictions))
        return {"job_id": job_id, "status": "pending"}

    @bentoml.api
    def status(self, job_id: str) -> dict:
        return self.jobs.get(job_id, {"job_id": job_id, "status": "not_found"})


@bentoml.service(name="admission_api", workers=1)
class AdmissionAPI:
    """The public service: routing, validation and JWT authentication. No model here."""

    single = (
        bentoml.depends(SinglePredictor, url=SINGLE_PREDICTOR_URL)
        if SINGLE_PREDICTOR_URL
        else bentoml.depends(SinglePredictor)
    )
    batch = (
        bentoml.depends(BatchPredictor, url=BATCH_PREDICTOR_URL)
        if BATCH_PREDICTOR_URL
        else bentoml.depends(BatchPredictor)
    )

    @bentoml.api(route="/login")
    def login(self, username: str, password: str) -> dict:
        if verify_credentials(username, password):
            return {"token": create_jwt_token(username)}
        return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})

    @bentoml.api(route="/predict")
    def predict(
        self,
        gre_score: int,
        toefl_score: int,
        university_rating: int,
        sop: float,
        lor: float,
        cgpa: float,
        research: int,
    ) -> dict:
        """Single prediction, routed to the single worker.

        The fields are declared one by one so the request body is a flat JSON object
        (that is what the tests send). They are validated by rebuilding
        an AdmissionInput, which carries the range checks.
        """
        try:
            student = AdmissionInput(
                gre_score=gre_score,
                toefl_score=toefl_score,
                university_rating=university_rating,
                sop=sop,
                lor=lor,
                cgpa=cgpa,
                research=research,
            )
        except ValidationError as exc:
            return JSONResponse(status_code=400, content={"detail": exc.errors(include_url=False)})

        return self.single.predict(student)

    @bentoml.api(route="/batch_predict")
    async def batch_predict(self, predictions: list[AdmissionInput]) -> dict:
        """Send a batch to the batch worker and return its job id.

        async because the worker's submit is async: calling it returns a coroutine
        that must be awaited, otherwise the response is not the job info.
        """
        return await self.batch.submit(AdmissionBatchInput(predictions=predictions))

    @bentoml.api(route="/batch_status")
    def batch_status(self, job_id: str) -> dict:
        result = self.batch.status(job_id)
        if result.get("status") == "not_found":
            return JSONResponse(
                status_code=404, content={"detail": f"Job {job_id} not found"}
            )
        return result


# The middleware is attached to the API service only - it is the single entry point.
AdmissionAPI.add_asgi_middleware(JWTAuthMiddleware)
