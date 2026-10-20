# Iris Classification API

A REST API that serves a **Random Forest** model trained on the classic Iris dataset. Built with **FastAPI**, it supports local execution and Docker deployment.

## Features

- Accepts four Iris flower measurements as JSON.
- Returns the predicted species, class index, class probabilities, and confidence.
- Provides a health-check endpoint to check whether the model is loaded.
- Offers interactive API documentation through Swagger UI at `/docs`.

## Project structure

```text
ml-api-deploy/
├── app/
│   ├── main.py
│   ├── model.py
│   └── schemas.py
├── models/
│   └── iris_model.pkl
├── tests/
│   ├── __init__.py
│   └── test_api.py
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
└── train_model.py
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Adarshdx/ml-api-deploy.git
cd ml-api-deploy
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/macOS**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Train the model

If the model file is missing or you want to retrain it, run:

```bash
python train_model.py
```

The script creates the `models/` directory if necessary and saves the trained model to `models/iris_model.pkl`.

## Run the API locally

Ensure the model file exists, then run:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API is available at `http://localhost:8000`.

Interactive documentation: `http://localhost:8000/docs`

## Run with Docker Compose

Make sure Docker Desktop is running, then execute:

```bash
docker compose up -d --build
```

To check the container status:

```bash
docker compose ps
```

To stop the application:

```bash
docker compose down
```

## API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Returns API information and available endpoints |
| `/health` | GET | Reports the API and model-loading status |
| `/predict` | POST | Predicts the Iris species |
| `/docs` | GET | Opens interactive API documentation |

### Prediction example

**Request — `POST /predict`**

```json
{
  "features": {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }
}
```

**Example response**

```json
{
  "prediction": 0,
  "class_name": "setosa",
  "probabilities": [1.0, 0.0, 0.0],
  "confidence": 1.0
}
```

The prediction is the class index, `class_name` is the predicted species, `probabilities` contains the class probabilities, and `confidence` is the highest probability.

If the model is not loaded, the prediction endpoint returns HTTP 503.

## Run automated tests

Install the development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

Run the tests:

```bash
python -m pytest -q
```

The test suite uses `pytest` and FastAPI's `TestClient`.

## Repository

GitHub: [ml-api-deploy](https://github.com/Deepak-IITian07/ml-api-deploy)