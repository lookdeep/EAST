#!/bin/bash
set -eux

MODELS_ROOT=${MODELS_ROOT:-"/tmp/models"}

docker run \
    --runtime=nvidia \
    --name tfserve_east_gpu \
    --publish 18511:8501 \
    --publish 18510:8500 \
    --mount type=bind,source="$MODELS_ROOT/east",target=/models/east \
    --env MODEL_NAME=east \
    --env CUDA_VISIBLE_DEVICES=1 \
    --tty \
    eldon/tensorflow-serving-gpu
