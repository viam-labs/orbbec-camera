#!/bin/bash
set -e

# Check if pyorbbecsdk directory exists
if [ ! -d "pyorbbecsdk" ]; then
  echo "Error: pyorbbecsdk directory not found."
  echo "Please run: git submodule update --init --recursive"
  exit 1
fi

sudo apt install -y cmake gcc

# Build pyorbbecsdk
cd pyorbbecsdk
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
# uv pip install wheel
python setup.py bdist_wheel

# Install the wheel
uv pip install dist/*.whl

echo "pyorbbecsdk built and installed successfully!"
cd ..
