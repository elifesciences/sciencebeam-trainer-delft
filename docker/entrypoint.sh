#!/bin/bash

set -e

GROBID_MODELS_DIRECTORY=/opt/grobid/grobid-home/models
EMBEDDING_REGISTRY_PATH="${PROJECT_FOLDER}/embedding-registry.json"

if [ ! -z "${OVERRIDE_MODELS}" ]; then
    if [ ! -d "${GROBID_MODELS_DIRECTORY}" ]; then
        echo "directory does not exist: ${GROBID_MODELS_DIRECTORY}"
        exit 1
    fi
    echo "installing models: ${OVERRIDE_MODELS}"
    python -m sciencebeam_trainer_delft.sequence_labelling.tools.install_models \
        --model-base-path=${GROBID_MODELS_DIRECTORY} \
        --install "${OVERRIDE_MODELS}" \
        --validate-pickles
fi

if [ "${DISABLE_LMDB_CACHE}" == "1" ]; then
    echo "disabling lmdb cache: ${EMBEDDING_REGISTRY_PATH}"
    python -m sciencebeam_trainer_delft.embedding \
        disable-lmdb-cache \
        --registry-path=${EMBEDDING_REGISTRY_PATH}
elif [ ! -z "${EMBEDDING_LMDB_PATH}" ]; then
    echo "setting embedding lmdb path to: ${EMBEDDING_LMDB_PATH}"
    python -m sciencebeam_trainer_delft.embedding \
        set-lmdb-path \
        --registry-path=${EMBEDDING_REGISTRY_PATH} \
        --lmdb-cache-path=${EMBEDDING_LMDB_PATH}
fi

exec $@
