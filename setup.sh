#!/usr/bin/env bash

set -euo pipefail

cd $(dirname $0)

# initialize submodule
./install_requirements.sh

VENV_NAME=".venv"

# Create a virtual environment to run our code
export PATH=$PATH:$HOME/.local/bin

if [ ! "$(command -v uv)" ]; then
  if [ ! "$(command -v curl)" ]; then
    echo "curl is required to install UV. please install curl on this system to continue."
    exit 1
  fi
  echo "Installing uv command"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if ! uv venv $VENV_NAME --python 3.10; then
  echo "unable to create required virtual environment"
  exit 1
fi

source "./$VENV_NAME/bin/activate"

if ! uv sync; then
  echo "unable to sync requirements to venv"
  exit 1
fi

# build and install pyorbbecksdk wheels
./build_sdk.sh
