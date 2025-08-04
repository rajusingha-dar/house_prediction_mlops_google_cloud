import os
import sys
import pytest
import joblib
from sklearn.base import is_regressor

# Add the 'src' directory to the Python path so we can import 'train_model'
# This is necessary because we're running the test from the root directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.train import train_model

def test_train_model_creates_artifact():
    """
    Tests the train_model function to ensure it runs and creates a model file.
    """
    # Define the expected path for the model artifact
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'house_price_model.joblib')

    # If a model file already exists, remove it before the test
    if os.path.exists(model_path):
        os.remove(model_path)

    # Assert that the model file does not exist before training
    assert not os.path.exists(model_path)

    # Run the training process
    try:
        train_model()
    except Exception as e:
        pytest.fail(f"train_model() raised an exception: {e}")

    # Assert that the model file now exists after training
    assert os.path.exists(model_path)

    # Assert that the created artifact can be loaded and is a valid regressor
    try:
        loaded_model = joblib.load(model_path)
        assert is_regressor(loaded_model)
    except Exception as e:
        pytest.fail(f"Failed to load or validate the saved model artifact: {e}")

    # Clean up the created model file after the test
    if os.path.exists(model_path):
        os.remove(model_path)

