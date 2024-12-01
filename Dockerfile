# Use a lightweight Python image
FROM python:3.11-slim

# Install system-level dependencies, including libGL
RUN apt-get update && apt-get install -y python3-dev gcc libffi-dev libssl-dev libgl1

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the entire application
COPY . .

# Expose the port your app runs on
EXPOSE 8000

# Start the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
