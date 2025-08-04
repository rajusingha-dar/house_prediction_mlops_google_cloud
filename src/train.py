import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import os
from google.cloud import storage
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def train_model():
    """
    This function trains the house price prediction model, saves it locally,
    and uploads it to Google Cloud Storage.
    """
    print("Starting model training...")

    # --- 1. Load Data ---
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'AmesHousing.csv')
    df = pd.read_csv(data_path)
    print("Data loaded successfully.")

    # --- 2. Split Data ---
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split complete.")

    # --- 3. Define Preprocessing ---
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns
    numerical_features = numerical_features.drop(['Order', 'PID'])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    print("Preprocessor defined.")

    # --- 4. Define Model (using best params from our notebook) ---
    best_params = {
        'colsample_bytree': 0.7, 'learning_rate': 0.05, 'max_depth': 4,
        'n_estimators': 500, 'subsample': 0.8
    }
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42, **best_params)

    # --- 5. Create and Train Full Pipeline ---
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', xgb_model)])
    final_model = TransformedTargetRegressor(regressor=model_pipeline, func=np.log1p, inverse_func=np.expm1)

    print("Training the final model...")
    final_model.fit(X_train, y_train)
    print("Model training complete.")

    # --- 6. Save Model Artifact Locally ---
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    local_model_path = os.path.join(model_dir, 'house_price_model.joblib')
    joblib.dump(final_model, local_model_path)
    print(f"Model saved locally to {local_model_path}")

    # --- 7. Upload Model to GCS ---
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    if not bucket_name:
        raise ValueError("GCS_BUCKET_NAME environment variable not set.")
        
    gcs_model_path = 'models/house_price_model.joblib'
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_model_path)

    blob.upload_from_filename(local_model_path)
    print(f"Model uploaded to gs://{bucket_name}/{gcs_model_path}")


if __name__ == '__main__':
    train_model()
