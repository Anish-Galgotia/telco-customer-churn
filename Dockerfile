# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ src/
COPY app/ app/
COPY models/ models/

# Expose Flask port
EXPOSE 5000

# Run Flask app
CMD ["python", "app/app.py"]
