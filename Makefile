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

DELFT_RUN = $(DOCKER_COMPOSE) run --rm delft
DELFT_DEV_RUN = $(DELFT_RUN)
PYTEST_WATCH = $(DELFT_RUN) pytest-watch
PYTHON = $(DELFT_RUN) python

JUPYTER_DOCKER_COMPOSE = NB_UID="$(NB_UID)" NB_GID="$(NB_GID)" $(DOCKER_COMPOSE)
JUPYTER_RUN = $(JUPYTER_DOCKER_COMPOSE) run --rm jupyter

NOTEBOOK_OUTPUT_FILE =

NB_UID = $(shell id -u)
NB_GID = $(shell id -g)

LIMIT = 10000
ARCHITECTURE = CustomBidLSTM_CRF
EMBEDDING = https://github.com/elifesciences/sciencebeam-models/releases/download/v0.0.1/glove.6B.50d.txt.gz
INPUT_PATH = https://github.com/elifesciences/sciencebeam-datasets/releases/download/v0.0.1/delft-grobid-header-060518.train.gz
WORD_LSTM_UNITS = 100
FEATURE_INDICES =
FEATURE_EMBEDDING_SIZE = 0
GROBID_TRAIN_ACTION = train

PYTEST_ARGS =
NOT_SLOW_PYTEST_ARGS = -m 'not slow'

ARGS =


.PHONY: build


venv-clean:
	@if [ -d "$(VENV)" ]; then \
		rm -rf "$(VENV)"; \
	fi


venv-create:
	virtualenv -p python3 $(VENV)


dev-install:
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements.cpu.txt
	$(PIP) install -r requirements.dev.txt
	$(PIP) install -r requirements.delft.txt --no-deps


dev-venv: venv-create dev-install


build:
	$(DOCKER_COMPOSE) build delft


shell:
	$(DELFT_RUN) bash


pylint:
	$(DELFT_DEV_RUN) pylint sciencebeam_trainer_delft setup.py


flake8:
	$(DELFT_DEV_RUN) flake8 sciencebeam_trainer_delft setup.py


pytest:
	$(DELFT_DEV_RUN) pytest -p no:cacheprovider $(PYTEST_ARGS)


pytest-not-slow:
	@$(MAKE) PYTEST_ARGS="$(PYTEST_ARGS) $(NOT_SLOW_PYTEST_ARGS)" pytest


.watch:
	$(PYTEST_WATCH) -- -p no:cacheprovider -p no:warnings $(PYTEST_ARGS)


watch-slow:
	@$(MAKE) .watch


watch:
	@$(MAKE) PYTEST_ARGS="$(PYTEST_ARGS) $(NOT_SLOW_PYTEST_ARGS)" .watch


test-setup-install:
	$(DELFT_RUN) python setup.py install


test: \
	flake8 \
	pylint \
	pytest \
	test-setup-install


grobid-train-header:
	$(PYTHON) -m sciencebeam_trainer_delft.grobid_trainer \
		header $(GROBID_TRAIN_ACTION) \
		--batch-size="$(BATCH_SIZE)" \
		--word-lstm-units="$(WORD_LSTM_UNITS)" \
		--max-sequence-length="$(MAX_SEQUENCE_LENGTH)" \
		--embedding="$(EMBEDDING)" \
		--architecture="$(ARCHITECTURE)" \
		--feature-indices="$(FEATURE_INDICES)" \
		--feature-embedding-size="$(FEATURE_EMBEDDING_SIZE)" \
		--max-epoch="$(MAX_EPOCH)" \
		--input="$(INPUT_PATH)" \
		--output="$(MODEL_OUTPUT)" \
		--limit="$(LIMIT)" \
		--checkpoint="$(CHECKPOINT_OUTPUT)" \
		$(ARGS)


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
