# Makefile for Virtual Board project

.PHONY: help install test train run clean setup dev

# Default target
help:
	@echo "Virtual Board - Available commands:"
	@echo ""
	@echo "  make install    - Install dependencies and setup project"
	@echo "  make setup      - Quick setup (install + create directories)"
	@echo "  make train      - Train ML models"
	@echo "  make test       - Run all tests"
	@echo "  make run        - Run the Virtual Board application"
	@echo "  make demo       - Run the basic demo"
	@echo "  make dev        - Setup development environment"
	@echo "  make clean      - Clean generated files"
	@echo "  make help       - Show this help message"

# Install dependencies
install:
	@echo "Installing Virtual Board..."
	python scripts/install.py

# Quick setup
setup: install
	@echo "Setup completed!"

# Train ML models
train:
	@echo "Training ML models..."
	python scripts/train_models.py

# Train specific model
train-digit:
	@echo "Training digit recognition model..."
	python scripts/train_models.py --model digit

train-letter:
	@echo "Training letter recognition model..."
	python scripts/train_models.py --model letter

# Run tests
test:
	@echo "Running tests..."
	python scripts/run_tests.py

# Run main application
run:
	@echo "Starting Virtual Board..."
	python -m src.main

# Run demo
demo:
	@echo "Starting Virtual Board demo..."
	python demo.py

# Development setup
dev: install
	@echo "Setting up development environment..."
	pip install pytest pytest-cov black flake8
	@echo "Development environment ready!"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	@echo "Cleanup completed!"

# Format code
format:
	@echo "Formatting code..."
	black src/ tests/ scripts/
	@echo "Code formatted!"

# Lint code
lint:
	@echo "Linting code..."
	flake8 src/ tests/ scripts/
	@echo "Linting completed!"

# Full development check
check: lint test
	@echo "All checks passed!"