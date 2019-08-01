#!/bin/bash
set -eux

# download east weights
gdown https://drive.google.com/a/lookdeep.health/uc?id=0B3APw5BZJ67ETHNPaU9xUkVoV0U
unzip east_icdar2015_resnet_v1_50_rbox.zip
rm east_icdar2015_resnet_v1_50_rbox.zip

curl http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz | tar -xz
