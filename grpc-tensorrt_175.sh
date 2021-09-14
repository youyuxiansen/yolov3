#!/bin/sh
export OPENBLAS_CORETYPE=ARMV8
cd /home/yousixia/project/yolov3 && rm grpc-tensorrt.log && touch grpc-tensorrt.log
cd /home/yousixia/project/yolov3 && nohup /home/yousixia/anaconda3/envs/tensorrt_yolov3/bin/python trt_grpc_detect_server.py --choose-config yolov5s_nano > grpc-tensorrt.log 2>&1 &
