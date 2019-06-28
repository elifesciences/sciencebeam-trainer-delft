DOCKER_COMPOSE_DEV = docker-compose
DOCKER_COMPOSE_CI = docker-compose -f docker-compose.yml
DOCKER_COMPOSE = $(DOCKER_COMPOSE_DEV)

VENV = venv
PIP = $(VENV)/bin/pip
PYTHON = $(VENV)/bin/python

BATCH_SIZE = 10
MAX_EPOCH = 1
MAX_SEQUENCE_LENGTH = 500
MODEL_OUTPUT =
CHECKPOINT_OUTPUT =

PROJECT_FOLDER = /opt/sciencebeam-trainer-delft

DELFT_RUN = $(DOCKER_COMPOSE) run --rm delft

JUPYTER_DOCKER_COMPOSE = NB_UID="$(NB_UID)" NB_GID="$(NB_GID)" $(DOCKER_COMPOSE)
JUPYTER_RUN = $(JUPYTER_DOCKER_COMPOSE) run --rm jupyter

NOTEBOOK_OUTPUT_FILE =

NB_UID = $(shell id -u)
NB_GID = $(shell id -g)

EMBEDDING = https://github.com/elifesciences/sciencebeam-models/releases/download/v0.0.1/glove.6B.50d.txt.gz


.PHONY: build


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
	$(DELFT_RUN) bash


pylint:
	$(DELFT_RUN) pylint sciencebeam_trainer_delft "$(PROJECT_FOLDER)/setup.py"


flake8:
	$(DELFT_RUN) flake8 sciencebeam_trainer_delft "$(PROJECT_FOLDER)/setup.py"


pytest:
	$(DELFT_RUN) bash -c 'cd "$(PROJECT_FOLDER)" && PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider'


watch:
	$(DELFT_RUN) bash -c 'cd "$(PROJECT_FOLDER)" && PYTHONDONTWRITEBYTECODE=1 pytest-watch -- -p no:cacheprovider'


test-setup-install:
	$(DELFT_RUN) python "$(PROJECT_FOLDER)/setup.py" install


test: \
	flake8 \
	pylint \
	pytest \
	test-setup-install


grobid-train-header:
	$(DOCKER_COMPOSE) run --rm delft \
		python -m sciencebeam_trainer_delft.grobid_trainer \
		header train \
		--batch-size="$(BATCH_SIZE)" \
		--max-sequence-length="$(MAX_SEQUENCE_LENGTH)" \
		--embedding="$(EMBEDDING)" \
		--max-epoch="$(MAX_EPOCH)" \
		--output="$(MODEL_OUTPUT)" \
		--checkpoint="$(CHECKPOINT_OUTPUT)"


update-test-notebook:
	$(JUPYTER_RUN) update-notebook-and-check-no-errors.sh \
		test.ipynb "$(NOTEBOOK_OUTPUT_FILE)"


update-test-notebook-temp:
	$(MAKE) NOTEBOOK_OUTPUT_FILE="/tmp/dummy.ipynb" update-test-notebook


jupyter-build:
	@if [ "$(NO_BUILD)" != "y" ]; then \
		$(JUPYTER_DOCKER_COMPOSE) build jupyter; \
	fi


jupyter-shell: jupyter-build
	$(JUPYTER_RUN) bash


jupyter-start: jupyter-build
	$(JUPYTER_DOCKER_COMPOSE) up -d jupyter


jupyter-logs:
	$(JUPYTER_DOCKER_COMPOSE) logs -f jupyter


jupyter-stop:
	$(JUPYTER_DOCKER_COMPOSE) down


ci-build-and-test:
	$(MAKE) DOCKER_COMPOSE="$(DOCKER_COMPOSE_CI)" build test jupyter-build update-test-notebook-temp


ci-clean:
	$(DOCKER_COMPOSE_CI) down -v
