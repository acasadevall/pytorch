#!/usr/bin/env bash

source $HOME/pytorch/venv/bin/activate
source /opt/intel/oneapi/setvars.sh

TORCH_INSTALL_PATH=$HOME/pytorch/install/standalone_xpu_vulkan \
PYTHONPATH=$HOME/pytorch:$PYTHONPATH \
python3 $HOME/pytorch/apps/test_inference.py