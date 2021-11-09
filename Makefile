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
RUN_PYTHON = $(DELFT_RUN) python
TRAINER_GROBID_RUN = $(DOCKER_COMPOSE) run --rm --no-deps trainer-grobid

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

PDF_DATA_DIR = /data/pdf
DATASET_DIR = /data/dataset
USER_AGENT = Dummy/user-agent
SAMPLE_PDF_URL = https://cdn.elifesciences.org/articles/32671/elife-32671-v2.pdf

GCLOUD = gcloud
GCLOUD_JOB_NAME = sciencebeam_$(GROBID_TRAIN_ACTION)_$(ARCHITECTURE)_$(LIMIT)_$(shell date +%s -u)
GCLOUD_JOB_DIR =
# see https://cloud.google.com/ml-engine/docs/tensorflow/runtime-version-list
GCLOUD_AI_PLATFORM_RUNTIME = 1.15
GCLOUD_AI_PLATFORM_PYTHON_VERSION = 3.7
GCLOUD_ARGS =

PYTEST_ARGS =
NOT_SLOW_PYTEST_ARGS = -m 'not slow'
SLOW_PYTEST_ARGS = -m 'slow'

SYSTEM_PYTHON = python3

ARGS =


.PHONY: build


venv-clean:
	@if [ -d "$(VENV)" ]; then \
		rm -rf "$(VENV)"; \
	fi


venv-create:
	$(SYSTEM_PYTHON) -m venv $(VENV)


dev-install:
	$(PIP) install -r requirements.build.txt
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements.cpu.txt
	$(PIP) install -r requirements.dev.txt
	$(PIP) install -r requirements.delft.txt --no-deps
	$(PIP) install -r requirements.jep.txt
	$(PIP) install -e . --no-deps


dev-venv: venv-create dev-install


dev-flake8:
	$(PYTHON) -m flake8 sciencebeam_trainer_delft tests setup.py


dev-pylint:
	$(PYTHON) -m pylint sciencebeam_trainer_delft tests setup.py


dev-mypy:
	$(PYTHON) -m mypy --ignore-missing-imports sciencebeam_trainer_delft tests setup.py $(ARGS)


dev-lint: dev-flake8 dev-pylint dev-mypy


dev-pytest:
	$(PYTHON) -m pytest -v -p no:cacheprovider $(ARGS)


dev-watch:
	$(PYTHON) -m pytest_watch -- -p no:cacheprovider -p no:warnings $(NOT_SLOW_PYTEST_ARGS) $(ARGS)


dev-watch-slow:
	$(PYTHON) -m pytest_watch -- -p no:cacheprovider -p no:warnings $(ARGS)


dev-test: dev-lint dev-pytest


dev-remove-dist:
	rm -rf ./dist


dev-build-dist:
	$(PYTHON) setup.py sdist bdist_wheel


build:
	$(DOCKER_COMPOSE) build delft


shell:
	$(DELFT_RUN) bash


shell-dev:
	$(DELFT_DEV_RUN) bash


pylint:
	$(DELFT_DEV_RUN) pylint sciencebeam_trainer_delft tests setup.py


flake8:
	$(DELFT_DEV_RUN) flake8 sciencebeam_trainer_delft tests setup.py


mypy:
	$(DELFT_DEV_RUN) mypy --ignore-missing-imports sciencebeam_trainer_delft tests setup.py


pytest:
	$(DELFT_DEV_RUN) pytest -v -p no:cacheprovider $(PYTEST_ARGS)


pytest-not-slow:
	@$(MAKE) PYTEST_ARGS="$(PYTEST_ARGS) $(NOT_SLOW_PYTEST_ARGS)" pytest


pytest-slow:
	@$(MAKE) PYTEST_ARGS="$(PYTEST_ARGS) $(SLOW_PYTEST_ARGS)" pytest


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
	pylint \
	mypy


test: \
	lint \
	pytest \
	test-setup-install


.grobid-common-args:
	$(eval _GROBID_COMMON_ARGS = \
		--batch-size="$(BATCH_SIZE)" \
		--max-sequence-length="$(MAX_SEQUENCE_LENGTH)" \
		--input=$(INPUT_PATHS) \
		--limit="$(LIMIT)" \
	)

.grobid-train-header-args: .grobid-common-args
	$(eval _GROBID_TRAIN_ARGS = \
		header $(GROBID_TRAIN_ACTION) \
		$(_GROBID_COMMON_ARGS) \
		--word-lstm-units="$(WORD_LSTM_UNITS)" \
		--embedding="$(EMBEDDING)" \
		--architecture="$(ARCHITECTURE)" \
		--features-indices="$(FEATURE_INDICES)" \
		--features-embedding-size="$(FEATURE_EMBEDDING_SIZE)" \
		--max-epoch="$(MAX_EPOCH)" \
		--model-path="$(MODEL_PATH)" \
		--output="$(MODEL_OUTPUT)" \
		--checkpoint="$(CHECKPOINT_OUTPUT)" \
		$(ARGS) \
	)

.grobid-eval-header-args: .grobid-common-args
	$(eval _GROBID_EVAL_ARGS = \
		header eval \
		$(_GROBID_COMMON_ARGS) \
		--model-path="$(MODEL_PATH)" \
		$(ARGS) \
	)


.grobid-tag-header-args: .grobid-common-args
	$(eval _GROBID_TAG_ARGS = \
		header tag \
		$(_GROBID_COMMON_ARGS) \
		--model-path="$(MODEL_PATH)" \
		$(ARGS) \
	)


grobid-train-header: .grobid-train-header-args
	$(RUN_PYTHON) -m sciencebeam_trainer_delft.sequence_labelling.grobid_trainer \
		$(_GROBID_TRAIN_ARGS)


grobid-train-eval-header: .grobid-train-header-args
	$(MAKE) GROBID_TRAIN_ACTION=train_eval grobid-train-header


grobid-eval-header: .grobid-eval-header-args
	$(RUN_PYTHON) -m sciencebeam_trainer_delft.sequence_labelling.grobid_trainer \
		$(_GROBID_EVAL_ARGS)


grobid-tag-header: .grobid-tag-header-args
	$(RUN_PYTHON) -m sciencebeam_trainer_delft.sequence_labelling.grobid_trainer \
		$(_GROBID_TAG_ARGS)


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
	$(GCLOUD) beta ai-platform jobs submit training \
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


trainer-grobid-build:
	@if [ "$(NO_BUILD)" != "y" ]; then \
		$(DOCKER_COMPOSE) build trainer-grobid-base trainer-grobid; \
	fi


trainer-grobid-shell:
	$(DOCKER_COMPOSE) run --rm --no-deps trainer-grobid bash $(ARGS)


get-example-data:
	$(TRAINER_GROBID_RUN) bash -c '\
		mkdir -p "$(PDF_DATA_DIR)" \
		&& curl --fail --show-error --connect-timeout 60 --user-agent "$(USER_AGENT)" --location \
			"$(SAMPLE_PDF_URL)" --silent -o "$(PDF_DATA_DIR)/sample.pdf" \
		&& ls -l "$(PDF_DATA_DIR)" \
		'


generate-grobid-training-data:
	$(TRAINER_GROBID_RUN) \
		generate-grobid-training-data.sh \
		"${PDF_DATA_DIR}" \
		"$(DATASET_DIR)"


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
		build test grobid-build trainer-grobid-build jupyter-build update-test-notebook-temp


ci-build-and-test-core:
	$(MAKE) DOCKER_COMPOSE="$(DOCKER_COMPOSE_CI)" build test


ci-build-core:
	$(MAKE) DOCKER_COMPOSE="$(DOCKER_COMPOSE_CI)" build


ci-lint:
	$(MAKE) DOCKER_COMPOSE="$(DOCKER_COMPOSE_CI)" lint


ci-pytest-not-slow:
	$(MAKE) DOCKER_COMPOSE="$(DOCKER_COMPOSE_CI)" pytest-not-slow


ci-pytest-slow:
	$(MAKE) DOCKER_COMPOSE="$(DOCKER_COMPOSE_CI)" pytest-slow


ci-test-setup-install:
	$(MAKE) DOCKER_COMPOSE="$(DOCKER_COMPOSE_CI)" test-setup-install


ci-push-testpypi:
	$(DOCKER_COMPOSE_CI) run --rm \
		-v $$PWD/.pypirc:/root/.pypirc \
		delft \
		./scripts/dev/push-testpypi-commit-version.sh "$(REVISION)"


ci-build-grobid:
	$(MAKE) DOCKER_COMPOSE="$(DOCKER_COMPOSE_CI)" grobid-build


ci-build-grobid-trainer:
	$(MAKE) DOCKER_COMPOSE="$(DOCKER_COMPOSE_CI)" trainer-grobid-build


ci-build-and-test-jupyter:
	$(MAKE) DOCKER_COMPOSE="$(DOCKER_COMPOSE_CI)" jupyter-build update-test-notebook-temp


ci-clean:
	$(DOCKER_COMPOSE_CI) down -v
