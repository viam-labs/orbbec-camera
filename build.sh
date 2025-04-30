#!/usr/bin/env bash

set -euo pipefail

cd $(dirname $0)

export PATH=$PATH:$HOME/.local/bin
VENV_NAME=".venv"

source "./$VENV_NAME/bin/activate"

uv build --wheel

cp ./pyorbbecsdk/dist/*.whl ./dist/
cp ./scripts/* ./dist/

tar -czvf archive.tar.gz ./dist/ meta.json first_run.sh run.sh
