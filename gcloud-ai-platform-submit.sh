#!/usr/bin/env bash

echo "args: $@"

JOB_PREFIX=sciencebeam_
GCLOUD_AI_PLATFORM_RUNTIME=1.15
GCLOUD_AI_PLATFORM_PYTHON_VERSION=3.7

GCLOUD_ARGS=()
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --job-prefix)
            JOB_PREFIX="$2"
            shift # past argument
            shift # past value
            ;;
        --)
            shift # past value
            break
            ;;
        *)
            GCLOUD_ARGS+=("$1") # save it in an array for later
            shift # past argument
            ;;
    esac
done

if [ -z "$1" ]; then
    set -- "${GCLOUD_ARGS[@]}" # restore positional parameters
    GCLOUD_ARGS=()
fi

echo "GCLOUD_ARGS: ${GCLOUD_ARGS[@]}"
echo "args: $@"

GCLOUD_JOB_NAME="${JOB_PREFIX}_`date +%s -u`"

echo "GCLOUD_JOB_NAME: ${GCLOUD_JOB_NAME}"

gcloud beta ai-platform jobs submit training \
    "${GCLOUD_JOB_NAME}" \
    --stream-logs \
    --module-name sciencebeam_trainer_delft.sequence_labelling.grobid_trainer \
    --package-path sciencebeam_trainer_delft \
    --runtime-version "${GCLOUD_AI_PLATFORM_RUNTIME}" \
    --python-version "${GCLOUD_AI_PLATFORM_PYTHON_VERSION}" \
    "${GCLOUD_ARGS[@]}" \
    -- \
    "$@"
