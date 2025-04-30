#!/bin/bash
set -e

# Check if pyorbbecsdk directory exists
if [ ! -d "pyorbbecsdk" ]; then
  echo "Error: pyorbbecsdk directory not found."
  echo "Please run: git submodule update --init --recursive"
  exit 1
fi

sudo apt install -y cmake gcc clang

uv pip install setuptools

# Build pyorbbecsdk
cd pyorbbecsdk
uv pip install -r requirements.txt
if [ -d "build" ]; then
  rm -rf build
fi
mkdir -p build
cd build
cmake -Dpybind11_DIR=$(pybind11-config --cmakedir) ..
make -j4
make install
cd ..

# Build wheel
python setup.py bdist_wheel

echo "pyorbbecsdk built successfully!"
cd ..
