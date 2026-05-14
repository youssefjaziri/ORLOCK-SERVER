FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src ./src
COPY setup.py .

# Install the package
RUN pip install -e .

# Create directories for audio storage
RUN mkdir -p tempaudio transcriptions

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.orlock.main:app", "--host", "0.0.0.0", "--port", "8000"]
