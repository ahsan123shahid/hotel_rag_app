# --- Stage 1: Build React Frontend ---
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# --- Stage 2: Final lightweight image ---
FROM python:3.11-slim

# Install system dependencies needed for compiling python dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (Hugging Face Spaces requires UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy python backend code
COPY --chown=user hotel_rag_app/ hotel_rag_app/

# Copy compiled React frontend assets from Stage 1
COPY --chown=user --from=frontend-builder /app/frontend/dist/ frontend/dist/

# Set working directory to the flask code directory
WORKDIR $HOME/app/hotel_rag_app

# Expose default port (Hugging Face default is 7860)
EXPOSE 7860

# Run Flask app (which dynamically binds to $PORT, defaulting to 7860 on Spaces)
CMD ["python", "app.py"]
