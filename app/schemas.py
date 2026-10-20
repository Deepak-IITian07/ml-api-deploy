from pydantic import BaseModel, Field
from typing import List, Optional

class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=0, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, description="Petal length in cm")
    petal_width: float = Field(..., ge=0, description="Petal width in cm")
    
    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

class PredictionResponse(BaseModel):
    prediction: int
    class_name: str
    probabilities: List[float]
    confidence: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 0,
                "class_name": "setosa",
                "probabilities": [0.95, 0.03, 0.02],
                "confidence": 0.95
            }
        }

class PredictionRequest(BaseModel):
    features: IrisFeatures
