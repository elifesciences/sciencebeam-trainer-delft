#!/bin/bash

set -e

notebook_file="$1"


if [ -z "${notebook_file}" ]; then
    echo "Usage: $0 <notebook_file>"
    exit 1
fi


stderr_content=$(
    cat "$notebook_file" \
    | jq --raw-output \
    '.cells[].outputs[]? | select(.name == "stderr") | .text[]'
)
if [ ! -z "$stderr_content" ]; then
    echo -e "Error: Notebook contains stderr output: >>>\n$stderr_content\n<<<"
    exit 3
fi
