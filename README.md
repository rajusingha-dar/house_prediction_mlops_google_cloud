End-to-End House Price Prediction with MLOps on Google Cloud
This project demonstrates a full end-to-end machine learning workflow for predicting house prices using the Ames Housing dataset. It covers everything from data exploration and model training to deploying a scalable prediction API on Google Cloud Platform (GCP) with a complete CI/CD pipeline.

Project Overview
The core goal is to build a reliable and automated system that:

Trains a machine learning model to predict house sale prices.

Experiments with multiple algorithms to find the best-performing model.

Exposes the champion model through a REST API built with FastAPI.

Containerizes the application using Docker.

Automates the deployment process to Google Cloud Run using a CI/CD pipeline with GitHub Actions.

Tech Stack
Backend: Python, FastAPI

ML & Data Science: Scikit-learn, Pandas, XGBoost, SHAP, Joblib

Cloud Platform: Google Cloud Platform (GCP)

Key GCP Services: Vertex AI, Cloud Run, Artifact Registry, Cloud Build

CI/CD & Version Control: Git, GitHub, GitHub Actions

Containerization: Docker

Project Structure
house-price-prediction/
│
├── .github/workflows/      # CI/CD pipeline for deploying the API
│   └── main.yml
│
├── app/                    # FastAPI application code
│   └── main.py
│
├── data/                   # Raw data for local training
│   └── AmesHousing.csv
│
├── models/                 # Saved model artifacts
│   └── house_price_model.joblib
│
├── src/                    # Core ML source code
│   ├── train.py            # Model training script
│   └── predict.py          # Prediction logic
│
├── tests/                  # Unit and integration tests
│   └── test_training.py
│
├── .gitignore              # Files to ignore in Git
├── Dockerfile              # Instructions to build the app container
└── requirements.txt        # Python dependencies

Local Setup and Installation
Follow these steps to set up and run the project on your local machine.

1. Clone the Repository (if applicable)

git clone https://github.com/YOUR_USERNAME/house_prediction_mlops_google_cloud.git
cd house_prediction_mlops_google_cloud

2. Create and Activate a Virtual Environment
It is highly recommended to use a virtual environment to manage project dependencies.

# Create the environment
python -m venv venv

# Activate it (macOS/Linux)
source venv/bin/activate

# Activate it (Windows)
.\venv\Scripts\activate

3. Install Dependencies
Install all the required Python libraries.

pip install -r requirements.txt

Usage
1. Train the Model
Run the training script. This will process the data and save the trained model artifact to the models/ directory.

python src/train.py

2. Run Tests
Verify that the training process works as expected.

pytest

3. Run the FastAPI Server Locally
Start the local web server to serve your prediction API.

uvicorn app.main:app --reload

The API will be available at http://127.0.0.1:8000. You can access the interactive documentation at http://127.0.0.1:8000/docs.

API Endpoint
URL: /predict

Method: POST

Description: Receives key house features and returns a predicted price.

Example Payload:

{
  "GrLivArea": 1710,
  "OverallQual": 7,
  "YearBuilt": 2003,
  "TotalBsmtSF": 856,
  "GarageCars": 2,
  "FullBath": 2,
  "Neighborhood": "CollgCr"
}

Success Response:

{
  "predicted_price": 205495.31
}

Deployment to Google Cloud (CI/CD)
This project is configured for automatic deployment to Google Cloud Run using GitHub Actions.

The workflow is defined in .github/workflows/main.yml. It triggers on every push to the main branch and performs the following steps:

Authenticates to Google Cloud using a Service Account.

Builds a Docker container image of the FastAPI application.

Pushes the container image to Google Artifact Registry.

Deploys the new image to the Cloud Run service.

Required GitHub Secrets
To enable the CI/CD pipeline, you must configure the following secrets in your GitHub repository settings (Settings > Secrets and variables > Actions):

GCP_PROJECT_ID: Your Google Cloud Project ID.


GCP_SA_KEY: The JSON key for the Google Cloud Service Account you created with the necessary permissions.