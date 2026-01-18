#!/bin/bash
# Bio-RAG Production Deployment Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Bio-RAG Deployment Script"
echo "=========================================="

# Check if .env exists
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo "ERROR: .env file not found!"
    echo "Please create .env file from .env.example"
    exit 1
fi

# Load environment variables
source "$PROJECT_DIR/.env"

# Validate required environment variables
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
    echo "ERROR: OPENAI_API_KEY is not set in .env"
    exit 1
fi

if [ -z "$POSTGRES_PASSWORD" ] || [ "$POSTGRES_PASSWORD" = "your_secure_password_here" ]; then
    echo "ERROR: POSTGRES_PASSWORD is not set in .env"
    exit 1
fi

if [ -z "$JWT_SECRET_KEY" ] || [ "$JWT_SECRET_KEY" = "your_jwt_secret_key_here" ]; then
    echo "ERROR: JWT_SECRET_KEY is not set in .env"
    exit 1
fi

echo "[1/5] Pulling latest code..."
cd "$PROJECT_DIR"
git pull origin main

echo "[2/5] Building Docker images..."
docker compose -f deploy/docker-compose.prod.yml build --no-cache

echo "[3/5] Stopping existing containers..."
docker compose -f deploy/docker-compose.prod.yml down

echo "[4/5] Starting services..."
docker compose -f deploy/docker-compose.prod.yml up -d

echo "[5/5] Waiting for services to be healthy..."
sleep 30

# Health check
echo ""
echo "Checking service health..."
if curl -s http://localhost/health | grep -q "healthy"; then
    echo "✅ Backend is healthy"
else
    echo "⚠️ Backend health check failed"
fi

if curl -s -o /dev/null -w "%{http_code}" http://localhost | grep -q "200\|301\|302"; then
    echo "✅ Frontend is accessible"
else
    echo "⚠️ Frontend health check failed"
fi

echo ""
echo "=========================================="
echo "Deployment complete!"
echo ""
echo "Services:"
docker compose -f deploy/docker-compose.prod.yml ps
echo ""
echo "Logs: docker compose -f deploy/docker-compose.prod.yml logs -f"
echo "=========================================="
