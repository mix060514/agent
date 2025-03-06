# Use a multi-stage build for a smaller final image
FROM python:3.12-slim AS builder

# Install UV directly
RUN pip install uv

# Set UV environment variables for optimal Docker usage
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Copy dependency files first to leverage Docker caching
COPY agent/pyproject.toml agent/uv.lock ./

# Install dependencies directly with pip
RUN pip install -e .

# Copy the rest of the application code
COPY agent/ ./

# Create a smaller runtime image
FROM python:3.12-slim

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy the installed packages and application
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /app /app

# Switch to non-root user
USER appuser
WORKDIR /app

# Set the entry point to run the main.py file
CMD ["python", "main.py"]
