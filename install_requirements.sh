#!/bin/bash
set -e

# Initialize and update the git submodule
git submodule update --init --recursive

# Check if the submodule is on the correct branch
cd pyorbbecsdk
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "v2-main" ]; then
  echo "Switching pyorbbecsdk to v2-main branch"
  git checkout v2-main
fi
cd ..
