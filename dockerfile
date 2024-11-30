# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set a working directory inside the container
WORKDIR /app

# Copy the Flask app and model files into the container
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && apt-get clean

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask will run on
EXPOSE 5000

# Set the command to run the Flask app
CMD ["python", "main.py"]
