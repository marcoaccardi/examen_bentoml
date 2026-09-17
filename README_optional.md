# Admission Prediction Service — Optional Part

The same model, served as three services instead of one:

```
┌─────────────┐      ┌──────────────┐      ┌──────────────────┐
│ Client HTTP │─────>│ AdmissionAPI │─────>│ SinglePredictor  │
│   (tests)   │      │   :3000      │      │      :3001       │
└─────────────┘      │              │      └──────────────────┘
                     │              │      ┌──────────────────┐
                     │              │─────>│ BatchPredictor   │
                     │              │      │      :3002       │
                     └──────────────┘      └──────────────────┘
```

- **`AdmissionAPI`**: the public service. Routing, validation and JWT authentication.
- **`SinglePredictor`**: one prediction per request.
- **`BatchPredictor`**: takes a batch, returns a `job_id` and computes in the background.

Requires Docker (with Compose) and Python 3. Run the commands in order, from this folder.

## 1. Train and register the model

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

mkdir -p data/raw data/processed
curl -L -o data/raw/admission.csv \
  https://assets-datascientest.s3.eu-west-1.amazonaws.com/MLOPS/bentoml/admission.csv

python src/prepare_data.py     # data/raw/admission.csv -> data/processed/*.csv
python src/train_model.py      # trains the model and saves it as admission_lr
bentoml models list            # admission_lr must be in the list
```

## 2. Build the bento

```bash
bentoml build -f bentofile.yml
bentoml list
```

## 3. Build the Docker image

```bash
bentoml containerize admission_api:latest -t admission_service:latest
```

## 4. Run the architecture

```bash
docker compose up -d
docker compose ps          # api, single and batch must all be running
```

## 5. Run the tests

```bash
pytest -v tests/test_endpoints.py
```

## 6. Stop everything

```bash
docker compose down
```

## API reference

Request bodies are flat JSON objects. Every endpoint except `/login` needs the header
`Authorization: Bearer <token>`.

### `POST /login`

```bash
curl -X POST http://localhost:3000/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user123", "password": "password123"}'
```

```json
{"token": "eyJhbGciOi..."}
```

### `POST /predict`

```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" \
  -d '{"gre_score": 309, "toefl_score": 108, "university_rating": 4,
       "sop": 3.0, "lor": 4.0, "cgpa": 7.94, "research": 0}'
```

```json
{"prediction": [0.6324819623435172]}
```

### `POST /batch_predict`

```bash
curl -X POST http://localhost:3000/batch_predict \
  -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" \
  -d '{"predictions": [{"gre_score": 309, "toefl_score": 108, "university_rating": 4,
                        "sop": 3.0, "lor": 4.0, "cgpa": 7.94, "research": 0}]}'
```

```json
{"job_id": "0f0e...", "status": "pending"}
```

### `POST /batch_status`

```bash
curl -X POST http://localhost:3000/batch_status \
  -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" \
  -d '{"job_id": "0f0e..."}'
```

```json
{"job_id": "0f0e...", "status": "completed", "predictions": [0.632]}
```

Job statuses are `pending`, `completed` and `failed`. An unknown id returns
`404 {"detail": "Job <id> not found"}`.

## Project layout

```
src/
├── auth/jwt_auth.py       # token creation, credential check, JWT middleware
├── models/input_model.py  # AdmissionInput, AdmissionBatchInput, Credentials
├── prepare_data.py        # cleans and splits the data
├── train_model.py         # trains and saves the model
└── service_batch.py       # SinglePredictor, BatchPredictor, AdmissionAPI
tests/test_endpoints.py    # endpoint tests
bentofile.yml              # bento configuration
docker-compose.yml         # three containers from one image
```
