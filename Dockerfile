
FROM python:3.12-slim 

# Set the working directory
WORKDIR /app

# Install the OpenMP library required by XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install project dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Copy application files and saved model
COPY serve.py predict.py ./
COPY models ./models

# Expose the API port
EXPOSE 9696

# Start the API server
CMD ["uv", "run", "--no-sync", "uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "9696"]
