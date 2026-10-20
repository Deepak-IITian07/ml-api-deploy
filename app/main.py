# app/main.py – cleaned up and ready for deployment

from fastapi import FastAPI, HTTPException
from app.schemas import PredictionResponse, PredictionRequest
from app.model import classifier

# Create the FastAPI app with metadata that appears in the docs
app = FastAPI(
    title="Iris Classification API",
    description="API for predicting Iris flower species using Random Forest",
    version="1.0.0",
)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to Iris Classification API",
        "endpoints": {
            "/predict": "POST - Make a prediction",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation",
        },
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": classifier.model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction for Iris flower species."""
    if classifier.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    features_list = [
        request.features.sepal_length,
        request.features.sepal_width,
        request.features.petal_length,
        request.features.petal_width,
    ]
    return classifier.predict(features_list)
