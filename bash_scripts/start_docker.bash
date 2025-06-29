#!/bin/bash

docker run -it \
    --volume="./:/SplaTAM/" \
    -v /data/xthuang/dataset/slam/Replica:/SplaTAM/data/Replica \
    --env="NVIDIA_VISIBLE_DEVICES=all" \
    --env="NVIDIA_DRIVER_CAPABILITIES=all" \
    --net=host \
    --privileged \
    --group-add audio \
    --group-add video \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --name splatam_xiong \
    --gpus all \
    nkeetha/splatam:v1 \
    /bin/bash
    
