DOCKER_COMPOSE_DEV = docker-compose
DOCKER_COMPOSE_CI = docker-compose -f docker-compose.yml
DOCKER_COMPOSE = $(DOCKER_COMPOSE_DEV)

VENV = venv
PIP = $(VENV)/bin/pip
PYTHON = $(VENV)/bin/python

BATCH_SIZE = 10
MAX_EPOCH = 1
MAX_SEQUENCE_LENGTH = 500
MODEL_PATH =
MODEL_OUTPUT =
CHECKPOINT_OUTPUT =

DELFT_RUN = $(DOCKER_COMPOSE) run --rm delft
DELFT_DEV_RUN = $(DELFT_RUN)
PYTEST_WATCH = $(DELFT_DEV_RUN) pytest-watch
PYTHON = $(DELFT_RUN) python

JUPYTER_DOCKER_COMPOSE = NB_UID="$(NB_UID)" NB_GID="$(NB_GID)" $(DOCKER_COMPOSE)
JUPYTER_RUN = $(JUPYTER_DOCKER_COMPOSE) run --rm jupyter

NOTEBOOK_OUTPUT_FILE =

NB_UID = $(shell id -u)
NB_GID = $(shell id -g)

LIMIT = 10000
ARCHITECTURE = CustomBidLSTM_CRF
EMBEDDING = glove.6B.50d
INPUT_PATH = https://github.com/elifesciences/sciencebeam-datasets/releases/download/v0.0.1/delft-grobid-header-060518.train.gz
INPUT_PATHS = "$(INPUT_PATH)"
WORD_LSTM_UNITS = 100
FEATURE_INDICES =
FEATURE_EMBEDDING_SIZE = 0
GROBID_TRAIN_ACTION = train

GCLOUD = gcloud
GCLOUD_JOB_NAME = sciencebeam_$(GROBID_TRAIN_ACTION)_$(ARCHITECTURE)_$(LIMIT)_$(shell date +%s -u)
GCLOUD_JOB_DIR =
# see https://cloud.google.com/ml-engine/docs/tensorflow/runtime-version-list
GCLOUD_AI_PLATFORM_RUNTIME = 1.13
GCLOUD_AI_PLATFORM_PYTHON_VERSION = 3.5
GCLOUD_ARGS =

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
	$(PIP) install -r requirements.jep.txt
	$(PIP) install -e . --no-deps


dev-venv: venv-create dev-install


build:
	$(DOCKER_COMPOSE) build delft


shell:
	$(DELFT_RUN) bash


shell-dev:
	$(DELFT_DEV_RUN) bash


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


lint: \
	flake8 \
	pylint


test: \
	lint \
	pytest \
	test-setup-install


.grobid-train-header-args:
	$(eval _GROBID_TRAIN_ARGS = \
		header $(GROBID_TRAIN_ACTION) \
		--batch-size="$(BATCH_SIZE)" \
		--word-lstm-units="$(WORD_LSTM_UNITS)" \
		--max-sequence-length="$(MAX_SEQUENCE_LENGTH)" \
		--embedding="$(EMBEDDING)" \
		--architecture="$(ARCHITECTURE)" \
		--feature-indices="$(FEATURE_INDICES)" \
		--feature-embedding-size="$(FEATURE_EMBEDDING_SIZE)" \
		--max-epoch="$(MAX_EPOCH)" \
		--input=$(INPUT_PATHS) \
		--model-path="$(MODEL_PATH)" \
		--output="$(MODEL_OUTPUT)" \
		--limit="$(LIMIT)" \
		--checkpoint="$(CHECKPOINT_OUTPUT)" \
		$(ARGS) \
	)


grobid-train-header: .grobid-train-header-args
	$(PYTHON) -m sciencebeam_trainer_delft.sequence_labelling.grobid_trainer \
		$(_GROBID_TRAIN_ARGS)


grobid-eval-header:
	$(MAKE) GROBID_TRAIN_ACTION=eval grobid-train-header


grobid-tag-header:
	$(MAKE) GROBID_TRAIN_ACTION=tag grobid-train-header


gcloud-ai-platform-local-grobid-train-header: .grobid-train-header-args
	@echo "_GROBID_TRAIN_ARGS=$(_GROBID_TRAIN_ARGS)"
	$(GCLOUD) ai-platform local train \
		--module-name sciencebeam_trainer_delft.sequence_labelling.grobid_trainer \
		--package-path sciencebeam_trainer_delft \
		$(GCLOUD_ARGS) \
		-- \
		$(_GROBID_TRAIN_ARGS)


.require-GCLOUD_JOB_DIR:
	@if [ -z "$(GCLOUD_JOB_DIR)" ]; then \
		echo "GCLOUD_JOB_DIR required"; \
		exit 1; \
	fi


gcloud-ai-platform-cloud-grobid-train-header: .grobid-train-header-args .require-GCLOUD_JOB_DIR
	@echo "_GROBID_TRAIN_ARGS=$(_GROBID_TRAIN_ARGS)"
	@echo "GCLOUD_JOB_NAME=$(GCLOUD_JOB_NAME)"
	$(GCLOUD) ai-platform jobs submit training \
		"$(GCLOUD_JOB_NAME)" \
		--stream-logs \
		--job-dir "$(GCLOUD_JOB_DIR)" \
		--runtime-version "$(GCLOUD_AI_PLATFORM_RUNTIME)" \
		--python-version "$(GCLOUD_AI_PLATFORM_PYTHON_VERSION)" \
		--module-name sciencebeam_trainer_delft.sequence_labelling.grobid_trainer \
		--package-path sciencebeam_trainer_delft \
		$(GCLOUD_ARGS) \
		-- \
		$(_GROBID_TRAIN_ARGS)


grobid-build:
	@if [ "$(NO_BUILD)" != "y" ]; then \
		$(DOCKER_COMPOSE) build grobid; \
	fi


grobid-shell:
	$(DOCKER_COMPOSE) run --rm grobid bash


grobid-exec:
	$(DOCKER_COMPOSE) exec grobid bash


grobid-start: grobid-build
	$(DOCKER_COMPOSE) up -d grobid


grobid-logs:
	$(DOCKER_COMPOSE) logs -f grobid


grobid-stop:
	$(DOCKER_COMPOSE) stop grobid


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
	$(MAKE) DOCKER_COMPOSE="$(DOCKER_COMPOSE_CI)" \
		build test grobid-build jupyter-build update-test-notebook-temp


ci-clean:
	$(DOCKER_COMPOSE_CI) down -v
