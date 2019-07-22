#!/bin/bash

GROBID_MODELS_DIRECTORY=/opt/grobid/grobid-home/models

if [ ! -z "${OVERRIDE_MODELS}" ]; then
    if [ ! -d "${GROBID_MODELS_DIRECTORY}" ]; then
        echo "directory does not exist: ${GROBID_MODELS_DIRECTORY}"
        exit 1
    fi
    echo "installing models: ${OVERRIDE_MODELS}"
    python -m sciencebeam_trainer_delft.sequence_labelling.tools.install_models \
        --model-base-path=${GROBID_MODELS_DIRECTORY} \
        --install "${OVERRIDE_MODELS}"
fi

exec $@
