#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/torch/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

cd /ai/lzy/LiveCompose
python -u Adacrop/src/trainer.py