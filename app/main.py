from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
import sys
import os

# Add the 'src' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.predict import make_prediction

# Initialize the FastAPI app
app = FastAPI(title="House Price Prediction API")

# Define a SIMPLIFIED input data model for the user-facing API
class HouseDataSimple(BaseModel):
    # This setting is for Pydantic V2 and replaces allow_population_by_field_name
    model_config = ConfigDict(populate_by_name=True)

    # Key features we will ask the user for
    GrLivArea: int = Field(..., example=1500, description="Above grade (ground) living area square feet")
    OverallQual: int = Field(..., example=7, description="Rates the overall material and finish of the house (1-10)")
    YearBuilt: int = Field(..., example=2005, description="Original construction date")
    TotalBsmtSF: Optional[float] = Field(1000.0, description="Total square feet of basement area")
    GarageCars: Optional[float] = Field(2.0, description="Size of garage in car capacity")
    FullBath: int = Field(2, description="Full bathrooms above grade")
    Neighborhood: str = Field("NAmes", description="Physical locations within Ames city limits")

@app.get("/")
def read_root():
    return {"message": "Welcome to the House Price Prediction API"}


@app.post("/predict")
def predict_price(house_data: HouseDataSimple):
    """
    Prediction endpoint. Receives key house features and returns a predicted price.
    """
    try:
        # Convert the Pydantic model to a dictionary
        input_dict = house_data.dict()
        
        # Get the prediction by passing the simplified dictionary
        prediction = make_prediction(input_dict)
        
        return {"predicted_price": prediction}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

