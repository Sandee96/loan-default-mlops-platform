"""
schemas.py

Pydantic request/response models for the loan default prediction API.
The request represents raw input features only - engineered features
are computed internally and should not be user-supplied.
"""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Raw input features required to make a prediction."""

    LIMIT_BAL: float = Field(..., description="Amount of given credit (NT dollar)", gt=0)
    SEX: int = Field(..., description="Gender: 1 = male, 2 = female", ge=1, le=2)
    EDUCATION: int = Field(
        ..., description="Education level: 1=grad school, 2=university, 3=high school, 4=others",
        ge=1, le=4,
    )
    MARRIAGE: int = Field(..., description="Marital status: 1=married, 2=single, 3=others", ge=1, le=3)
    AGE: int = Field(..., description="Age in years", ge=18, le=100)

    PAY_0: int = Field(..., description="Repayment status, most recent month (-1=paid duly, 1+=months delayed)")
    PAY_2: int = Field(..., description="Repayment status, 2 months ago")
    PAY_3: int = Field(..., description="Repayment status, 3 months ago")
    PAY_4: int = Field(..., description="Repayment status, 4 months ago")
    PAY_5: int = Field(..., description="Repayment status, 5 months ago")
    PAY_6: int = Field(..., description="Repayment status, 6 months ago")

    BILL_AMT1: float = Field(..., description="Bill statement amount, most recent month")
    BILL_AMT2: float = Field(..., description="Bill statement amount, 2 months ago")
    BILL_AMT3: float = Field(..., description="Bill statement amount, 3 months ago")
    BILL_AMT4: float = Field(..., description="Bill statement amount, 4 months ago")
    BILL_AMT5: float = Field(..., description="Bill statement amount, 5 months ago")
    BILL_AMT6: float = Field(..., description="Bill statement amount, 6 months ago")

    PAY_AMT1: float = Field(..., description="Previous payment amount, most recent month", ge=0)
    PAY_AMT2: float = Field(..., description="Previous payment amount, 2 months ago", ge=0)
    PAY_AMT3: float = Field(..., description="Previous payment amount, 3 months ago", ge=0)
    PAY_AMT4: float = Field(..., description="Previous payment amount, 4 months ago", ge=0)
    PAY_AMT5: float = Field(..., description="Previous payment amount, 5 months ago", ge=0)
    PAY_AMT6: float = Field(..., description="Previous payment amount, 6 months ago", ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "LIMIT_BAL": 200000,
                "SEX": 2,
                "EDUCATION": 2,
                "MARRIAGE": 1,
                "AGE": 35,
                "PAY_0": 0,
                "PAY_2": 0,
                "PAY_3": 0,
                "PAY_4": 0,
                "PAY_5": 0,
                "PAY_6": 0,
                "BILL_AMT1": 50000,
                "BILL_AMT2": 48000,
                "BILL_AMT3": 46000,
                "BILL_AMT4": 44000,
                "BILL_AMT5": 42000,
                "BILL_AMT6": 40000,
                "PAY_AMT1": 2000,
                "PAY_AMT2": 2000,
                "PAY_AMT3": 2000,
                "PAY_AMT4": 2000,
                "PAY_AMT5": 2000,
                "PAY_AMT6": 2000,
            }
        }


class PredictionResponse(BaseModel):
    """Prediction result returned to the client."""

    prediction: int = Field(..., description="Predicted class: 0 = no default, 1 = default")
    probability: float = Field(..., description="Predicted probability of default (0.0 to 1.0)")
    risk_level: str = Field(..., description="Risk category: Low, Medium, or High")
    model_version: int = Field(..., description="Version number of the model that produced this prediction")

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 0,
                "probability": 0.23,
                "risk_level": "Low",
                "model_version": 1,
            }
        }