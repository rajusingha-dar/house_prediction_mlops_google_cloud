# train.py
# Training script for house price prediction
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

def train_model():
    """
    This function trains the house price prediction model and saves it.
    """
    print("Starting model training...")

    # --- 1. Load Data ---
    # In a real MLOps pipeline, this path would likely be a GCS URI
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
    # These are placeholder best parameters. Replace with your actual best params.
    best_params = {
        'colsample_bytree': 0.7,
        'learning_rate': 0.05,
        'max_depth': 4,
        'n_estimators': 500,
        'subsample': 0.8
    }
    
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_jobs=-1, 
        random_state=42,
        **best_params
    )

    # --- 5. Create and Train Full Pipeline ---
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb_model)
    ])

    final_model = TransformedTargetRegressor(
        regressor=model_pipeline,
        func=np.log1p,
        inverse_func=np.expm1
    )

    print("Training the final model...")
    final_model.fit(X_train, y_train)
    print("Model training complete.")

    # --- 6. Save Model Artifact ---
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'house_price_model.joblib')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(final_model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train_model()
