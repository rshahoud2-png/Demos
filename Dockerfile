FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Create logs directory and set permissions
RUN mkdir -p logs && chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Set default port (Fly.io sets PORT=8080)
ENV PORT=8080
ENV ENV=production

# Expose port
EXPOSE 8080

# Run the application using shell form to expand $PORT env var
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
