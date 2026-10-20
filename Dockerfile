# Updated Dockerfile
# - Removed the requests dependency in the HEALTHCHECK command.
#   Uses the standard library `urllib.request` to perform the check.
# - Updated the Python base image to python:3.11-slim to match current requirements.
#
# The application itself does not require any additional packages for health check.
# The Dockerfile still copies the correct application folders and installs the
# runtime dependencies.
FROM python:3.11-slim

WORKDIR /app

# Build dependencies for optional native extensions (sklearn comes from wheels)
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and model artifacts
COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8000

# Health check using the standard library
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request, json; \
        r = urllib.request.urlopen('http://localhost:8000/health'); \
        data=json.load(r); \
        exit(0) if data.get('status')=='healthy' else exit(1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
