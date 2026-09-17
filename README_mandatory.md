# Admission Prediction Service — Mandatory Part

API that predicts a student's chance of admission, secured with JWT.
Requires Docker and Python 3. Run the commands in order, from this folder.

## 1. Load the Docker image

```bash
docker load -i docker_image/admission_prediction_service.tar
```

## 2. Run the service

```bash
docker run --rm -p 3000:3000 --name admission_service admission_prediction_service:latest
```

The API listens on http://localhost:3000. Keep this terminal open and use a second one.

## 3. Install the test dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Run the tests

```bash
pytest -v
```

## 5. Stop the service

```bash
docker stop admission_service
```

## Example requests

Login (users: `user123` / `password123`, `admin` / `admin123`):

```bash
curl -X POST http://localhost:3000/login \
  -H "Content-Type: application/json" \
  -d '{"credentials": {"username": "user123", "password": "password123"}}'
```

Prediction, with the token from the login:

```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"input_data": {"gre_score": 309, "toefl_score": 108, "university_rating": 4,
       "sop": 3.0, "lor": 4.0, "cgpa": 7.94, "research": 0}}'
```

```json
{"prediction": [0.6324819623435172]}
```
