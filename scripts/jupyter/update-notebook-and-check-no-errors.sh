#!/bin/bash

set -e

notebook_file="$1"
output_notebook_file="${2:-$notebook_file}"


if [ -z "${notebook_file}" ]; then
    echo "Usage: $0 <notebook_file>"
    exit 1
fi

echo "output_notebook_file=${output_notebook_file}"

jupyter nbconvert --to notebook --execute "${notebook_file}" --output "${output_notebook_file}"

$(dirname $0)/check-notebook-no-errors.sh "${output_notebook_file}"
