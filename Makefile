APP_NAME = speech-sep-asr-poc
PORT = 8080

# Build Docker image
build:
	docker build -t $(APP_NAME) .

# Run container (detached)
run:
	docker run -d --rm -p $(PORT):8080 --name $(APP_NAME) $(APP_NAME)

# Run container (interactive, logs attached)
run-dev:
	docker run -it --rm -p $(PORT):8080 --name $(APP_NAME) $(APP_NAME)

# Stop container
stop:
	docker stop $(APP_NAME) || true

# Clean up dangling images/containers
clean:
	docker system prune -f

# CLI helper (requires API running locally on port 8080)
cli:
	python cli/transcribe.py sample_audio/harvard.wav --lang en

# Run API locally without Docker
local:
	uvicorn src.app:app --host 0.0.0.0 --port $(PORT) --reload

# Run tests locally
test-local:
	pytest -q

# Run tests inside Docker container
test-docker: build
	docker run --rm $(APP_NAME) pytest -q

