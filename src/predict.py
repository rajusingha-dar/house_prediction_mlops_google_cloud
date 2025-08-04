import joblib
import pandas as pd
import os

# --- Load the model (Robust Path) ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
model_path = os.path.join(PROJECT_ROOT, 'models', 'house_price_model.joblib')

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Make sure you have run the training script.")

# --- Load the entire model pipeline ---
model = joblib.load(model_path)

# --- Extract the feature names the model was trained on ---
try:
    pipeline = model.regressor_
    preprocessor = pipeline.named_steps['preprocessor']
    model_columns = preprocessor.feature_names_in_
except Exception as e:
    raise RuntimeError("Could not extract feature names from the model pipeline.") from e

# --- Define Default Values for a Simplified API ---
DEFAULT_INPUT_DATA = {
    "Order": 1, "PID": 526301100, "MSSubClass": 60, "MSZoning": "RL",
    "LotFrontage": 65.0, "LotArea": 9500, "Street": "Pave", "Alley": None,
    "LotShape": "Reg", "LandContour": "Lvl", "Utilities": "AllPub",
    "LotConfig": "Inside", "LandSlope": "Gtl", "Neighborhood": "NAmes",
    "Condition1": "Norm", "Condition2": "Norm", "BldgType": "1Fam",
    "HouseStyle": "2Story", "OverallQual": 7, "OverallCond": 5,
    "YearBuilt": 2005, "YearRemodAdd": 2006, "RoofStyle": "Gable",
    "RoofMatl": "CompShg", "Exterior1st": "VinylSd", "Exterior2nd": "VinylSd",
    "MasVnrType": "None", "MasVnrArea": 0.0, "ExterQual": "Gd",
    "ExterCond": "TA", "Foundation": "PConc", "BsmtQual": "Gd",
    "BsmtCond": "TA", "BsmtExposure": "No", "BsmtFinType1": "GLQ",
    "BsmtFinSF1": 0.0, "BsmtFinType2": "Unf", "BsmtFinSF2": 0.0,
    "BsmtUnfSF": 0.0, "TotalBsmtSF": 1000.0, "Heating": "GasA",
    "HeatingQC": "Ex", "CentralAir": "Y", "Electrical": "SBrkr",
    "1st Flr SF": 1200, "2nd Flr SF": 0, "LowQualFinSF": 0,
    "GrLivArea": 1500, "BsmtFullBath": 0.0, "BsmtHalfBath": 0.0,
    "FullBath": 2, "HalfBath": 1, "BedroomAbvGr": 3, "KitchenAbvGr": 1,
    "KitchenQual": "Gd", "TotRmsAbvGrd": 6, "Functional": "Typ",
    "Fireplaces": 1, "FireplaceQu": "Gd", "GarageType": "Attchd",
    "GarageYrBlt": 2005.0, "GarageFinish": "RFn", "GarageCars": 2.0,
    "GarageArea": 500.0, "GarageQual": "TA", "GarageCond": "TA",
    "PavedDrive": "Y", "WoodDeckSF": 0, "OpenPorchSF": 40,
    "EnclosedPorch": 0, "3Ssn Porch": 0, "ScreenPorch": 0, "PoolArea": 0,
    "PoolQC": None, "Fence": None, "MiscFeature": None, "MiscVal": 0,
    "MoSold": 6, "YrSold": 2008, "SaleType": "WD ", "SaleCondition": "Normal"
}


def make_prediction(input_data):
    """
    Makes a prediction using a simplified input dictionary.
    It merges user input with defaults to create a full feature set.
    """
    try:
        # Create a full feature dictionary by starting with defaults
        full_input = DEFAULT_INPUT_DATA.copy()
        # Update it with the values provided by the user
        full_input.update(input_data)
        
        # Convert the full dictionary to a pandas DataFrame
        input_df = pd.DataFrame([full_input])
        
        # Reindex to ensure column order matches the model's expectation
        input_df = input_df.reindex(columns=model_columns)

        # The model pipeline handles all preprocessing and prediction
        prediction = model.predict(input_df)
        
        # Convert the numpy float to a standard Python float for JSON serialization
        return float(prediction[0])
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        raise
