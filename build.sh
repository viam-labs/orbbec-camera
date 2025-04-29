#!/usr/bin/env bash

set -euo pipefail

cd $(dirname $0)

export PATH=$PATH:$HOME/.local/bin
VENV_NAME=".venv"

source "./$VENV_NAME/bin/activate"

# if ! uv pip install pyinstaller -q; then
#   exit 1
# fi

uv build --wheel

cp ./pyorbbecsdk/dist/*.whl ./dist/

# uv run pyinstaller --onefile -p src src/main.py
tar -czvf archive.tar.gz ./dist/ meta.json first_run.sh run.sh
