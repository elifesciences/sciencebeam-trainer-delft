#!/bin/bash

set -e

GROBID_CONFIG=${GROBID_CONFIG:-/opt/grobid/grobid-home/config/grobid.properties}

PROP_NAME="$1"
PROP_VALUE="$2"

echo "setting $PROP_NAME to $PROP_VALUE.."

UPDATED_CONFIG="$(cat $GROBID_CONFIG | grep --invert-match $PROP_NAME)"$'\n'"$PROP_NAME=$PROP_VALUE"

echo "$UPDATED_CONFIG" > $GROBID_CONFIG
