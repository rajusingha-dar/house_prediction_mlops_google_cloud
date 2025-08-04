# Start with an official Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code into the container
# This includes the 'app' and 'src' directories
COPY ./app /app/app
COPY ./src /app/src

# We NO LONGER copy the models directory.
# The application will download the model from GCS on startup.

# Expose the port the app runs on.
# Note: This is just metadata. The important part is the CMD instruction.
EXPOSE 8080

# Command to run the application using uvicorn
# This is the key fix: We use the "shell form" of CMD so that the $PORT
# environment variable is correctly interpreted by the shell.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
