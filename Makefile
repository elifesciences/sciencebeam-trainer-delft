DOCKER_COMPOSE_DEV = docker-compose
DOCKER_COMPOSE_CI = docker-compose -f docker-compose.yml
DOCKER_COMPOSE = $(DOCKER_COMPOSE_DEV)

VENV = venv
PIP = $(VENV)/bin/pip
PYTHON = $(VENV)/bin/python

MAX_EPOCH = 1
MAX_SEQUENCE_LENGTH = 500
MODEL_OUTPUT =
CHECKPOINT_OUTPUT =


venv-clean:
	@if [ -d "$(VENV)" ]; then \
		rm -rf "$(VENV)"; \
	fi


venv-create:
	virtualenv -p python3 $(VENV)


dev-install:
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements.dev.txt


dev-venv: venv-create dev-install


build:
	$(DOCKER_COMPOSE) build delft


shell:
	$(DOCKER_COMPOSE) run --rm delft bash


pylint:
	$(DOCKER_COMPOSE) run --rm delft pylint sciencebeam_trainer_delft


test: pylint


grobid-train-header:
	$(DOCKER_COMPOSE) run --rm delft \
		python -m sciencebeam_trainer_delft.grobid_trainer \
		header train \
		--max-sequence-length="$(MAX_SEQUENCE_LENGTH)" \
		--max-epoch="$(MAX_EPOCH)" \
		--output="$(MODEL_OUTPUT)" \
		--checkpoint="$(CHECKPOINT_OUTPUT)"


ci-build-and-test:
	$(MAKE) DOCKER_COMPOSE="$(DOCKER_COMPOSE_CI)" build test


ci-clean:
	$(DOCKER_COMPOSE_CI) down -v
