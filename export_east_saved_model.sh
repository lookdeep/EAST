#!/bin/bash
set -eux

MODELS_ROOT=${MODELS_ROOT:-"/var/models"}

if [[ -f east_icdar2015_resnet_v1_50_rbox ]]; then
    gdown "https://drive.google.com/uc?id=0B3APw5BZJ67ETHNPaU9xUkVoV0U"
    unzip east_icdar2015_resnet_v1_50_rbox.zip
    rm east_icdar2015_resnet_v1_50_rbox.zip
fi

python export_freeze_graph.py $MODELS_ROOT/east/1
