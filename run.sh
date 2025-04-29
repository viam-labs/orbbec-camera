#!/usr/bin/env bash

set -euo pipefail

cd $(dirname $0)

# Create a virtual environment to run our code
VENV_NAME="${VIAM_MODULE_DATA}/.venv"
PYTHON="$VENV_NAME/bin/python"

export PATH=$PATH:$HOME/.local/bin
if ! uv venv $VENV_NAME --python 3.10; then
  echo "unable to create required virtual environment"
  exit 1
fi
source $VENV_NAME/bin/activate
uv pip install ./dist/pyorbbecsdk-*.whl
uv pip install ./dist/orbbec_camera-*.whl

# Be sure to use `exec` so that termination signals reach the python process,
# or handle forwarding termination signals manually
echo "Starting module..."
exec $PYTHON -m main $@
