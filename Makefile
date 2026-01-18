.PHONY: help setup dev test build clean docker-up docker-down

help:
	@echo "Bio-RAG Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Install all dependencies"
	@echo "  make setup-backend  - Install backend dependencies"
	@echo "  make setup-frontend - Install frontend dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make dev            - Run both backend and frontend"
	@echo "  make dev-backend    - Run backend server"
	@echo "  make dev-frontend   - Run frontend server"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up      - Start all services with Docker"
	@echo "  make docker-down    - Stop all Docker services"
	@echo "  make docker-logs    - Show Docker logs"
	@echo ""
	@echo "Database:"
	@echo "  make db-migrate     - Run database migrations"
	@echo "  make db-upgrade     - Upgrade database to latest"
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Run all tests"
	@echo "  make test-backend   - Run backend tests"
	@echo "  make test-frontend  - Run frontend tests"
	@echo ""
	@echo "Build:"
	@echo "  make build          - Build for production"
	@echo "  make clean          - Clean build artifacts"

# Setup
setup: setup-backend setup-frontend
	@echo "Setup complete!"

setup-backend:
	cd backend && python -m venv venv && \
	. venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt

setup-frontend:
	cd frontend && npm install

# Development
dev:
	@echo "Starting development servers..."
	@make -j2 dev-backend dev-frontend

dev-backend:
	cd backend && . venv/bin/activate && uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	cd frontend && npm run dev

# Docker
docker-up:
	docker-compose -f infra/docker/docker-compose.yml up -d

docker-down:
	docker-compose -f infra/docker/docker-compose.yml down

docker-logs:
	docker-compose -f infra/docker/docker-compose.yml logs -f

docker-build:
	docker-compose -f infra/docker/docker-compose.yml build

# Database
db-migrate:
	cd backend && . venv/bin/activate && alembic revision --autogenerate -m "$(msg)"

db-upgrade:
	cd backend && . venv/bin/activate && alembic upgrade head

db-downgrade:
	cd backend && . venv/bin/activate && alembic downgrade -1

# Testing
test: test-backend test-frontend
	@echo "All tests passed!"

test-backend:
	cd backend && . venv/bin/activate && pytest tests/ -v --cov=src --cov-report=term-missing

test-frontend:
	cd frontend && npm run test

# Linting
lint:
	cd backend && . venv/bin/activate && ruff check src/
	cd frontend && npm run lint

format:
	cd backend && . venv/bin/activate && ruff format src/
	cd frontend && npm run format

# Build
build: build-backend build-frontend
	@echo "Build complete!"

build-backend:
	cd backend && . venv/bin/activate && pip install build && python -m build

build-frontend:
	cd frontend && npm run build

# Clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "node_modules" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned!"

# Celery
celery-worker:
	cd backend && . venv/bin/activate && celery -A src.tasks.celery worker --loglevel=info

celery-beat:
	cd backend && . venv/bin/activate && celery -A src.tasks.celery beat --loglevel=info
