#!/bin/bash
# Install all dependencies.
# flash-attn must be compiled from source and requires --no-build-isolation
# so that it can find the already-installed torch and CUDA headers.

set -e

pip install -r requirements.txt --no-build-isolation
