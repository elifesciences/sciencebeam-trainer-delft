#!/bin/bash

set -e

GROBID_CONFIG=${GROBID_CONFIG:-/opt/grobid/grobid-home/config/grobid.properties}

if [ ! -f "${GROBID_CONFIG}" ]; then
  echo "grobid config not found: ${GROBID_CONFIG}"
  exit 2
fi

PROP_NAME="$1"
PROP_VALUE="$2"

echo "setting $PROP_NAME to $PROP_VALUE.."

# remove existing properties via grep and add a new line, setting the new value
UPDATED_CONFIG="$(cat $GROBID_CONFIG | grep --invert-match $PROP_NAME)"$'\n'"$PROP_NAME=$PROP_VALUE"

echo "$UPDATED_CONFIG" > $GROBID_CONFIG
