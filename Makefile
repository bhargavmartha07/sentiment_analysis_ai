# ---------------------------------------------------------------------------
# Convenience wrapper around docker compose.
#   make up      - build and start the whole stack
#   make test    - run the unit and integration suites inside the containers
# ---------------------------------------------------------------------------
.DEFAULT_GOAL := help
COMPOSE := docker compose

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: up
up: ## Build images and start API, worker, RabbitMQ and MongoDB
	$(COMPOSE) up -d --build

.PHONY: down
down: ## Stop the stack (volumes are kept)
	$(COMPOSE) down

.PHONY: clean
clean: ## Stop the stack and delete volumes (database is wiped)
	$(COMPOSE) down -v --remove-orphans

.PHONY: rebuild
rebuild: ## Force a no-cache rebuild of both images
	$(COMPOSE) build --no-cache

.PHONY: logs
logs: ## Tail logs from every service
	$(COMPOSE) logs -f

.PHONY: ps
ps: ## Show container status and health
	$(COMPOSE) ps

.PHONY: health
health: ## Print the health endpoints
	@curl -fsS http://localhost:8000/health | python -m json.tool
	@curl -fsS http://localhost:8000/health/ready | python -m json.tool

.PHONY: test-unit-api
test-unit-api: ## Run API unit tests inside the api container
	$(COMPOSE) exec api python -m pytest tests/unit/api_tests.py -v

.PHONY: test-unit-services
test-unit-services: ## Run service unit tests inside the worker container
	$(COMPOSE) exec worker python -m pytest tests/unit/service_tests.py -v

.PHONY: test-integration
test-integration: ## Run end-to-end integration tests inside the api container
	$(COMPOSE) exec api python -m pytest tests/integration/integration_tests.py -v

.PHONY: test
test: test-unit-api test-unit-services test-integration ## Run every suite

.PHONY: coverage
coverage: ## Run unit tests with a coverage report
	$(COMPOSE) exec api python -m pytest tests/unit --cov=app --cov-report=term-missing

.PHONY: smoke
smoke: ## Submit sync + async jobs and print the results
	$(COMPOSE) exec api python scripts/smoke_test.py

.PHONY: benchmark
benchmark: ## Measure sync latency and throughput
	$(COMPOSE) exec api python scripts/benchmark.py

.PHONY: train
train: ## Retrain the Keras model and write artefacts into app/models (~3 min on CPU)
	$(COMPOSE) run --rm --no-deps -v "$(PWD)/app/models:/app/app/models" \
		api python scripts/train_model.py --epochs 5

.PHONY: rabbitmq-ui
rabbitmq-ui: ## Open the RabbitMQ management UI
	@echo "http://localhost:$${RABBITMQ_MANAGEMENT_PORT:-15672} (guest/guest)"
