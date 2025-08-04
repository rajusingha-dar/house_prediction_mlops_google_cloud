# Start with an official Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code into the container
# This includes the 'app', 'src', and 'models' directories
COPY ./app /app/app
COPY ./src /app/src
COPY ./models /app/models

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application using uvicorn
# This will start the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
