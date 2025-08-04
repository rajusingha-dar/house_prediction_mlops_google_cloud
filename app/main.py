from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
import sys
import os
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from .env file for local development
load_dotenv()

# Add the 'src' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.predict import make_prediction

# Initialize the FastAPI app
app = FastAPI(title="House Price Prediction API")

# --- CORS Configuration ---
# Load the allowed origins from an environment variable.
# The variable should contain a comma-separated list of URLs.
allowed_origins_str = os.getenv("ALLOWED_ORIGINS")
origins = []
if allowed_origins_str:
    origins = [origin.strip() for origin in allowed_origins_str.split(',')]
else:
    # Default to an empty list if the variable is not set
    print("WARNING: ALLOWED_ORIGINS environment variable not set. CORS will be restrictive.")


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define a SIMPLIFIED input data model for the user-facing API
class HouseDataSimple(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    GrLivArea: int = Field(..., example=1500)
    OverallQual: int = Field(..., example=7)
    YearBuilt: int = Field(..., example=2005)
    TotalBsmtSF: Optional[float] = Field(1000.0)
    GarageCars: Optional[float] = Field(2.0)
    FullBath: int = Field(2)
    Neighborhood: str = Field("NAmes")

@app.get("/")
def read_root():
    return {"message": "Welcome to the House Price Prediction API"}


@app.post("/predict")
def predict_price(house_data: HouseDataSimple):
    """
    Prediction endpoint. Receives key house features and returns a predicted price.
    """
    try:
        input_dict = house_data.dict()
        prediction = make_prediction(input_dict)
        return {"predicted_price": prediction}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
