#!/usr/bin/env bash

set -euo pipefail

cd $(dirname $0)

# Install udev rules
sudo bash ./dist/install_udev_rules.sh

VENV_NAME="${VIAM_MODULE_DATA}/.venv"

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

source "$VENV_NAME/bin/activate"

# if ! uv pip install -r requirements.txt; then
#   echo "unable to sync requirements to venv"
#   exit 1
# fi

# uv pip install
