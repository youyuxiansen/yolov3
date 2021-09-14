#!/bin/sh
export OPENBLAS_CORETYPE=ARMV8
cd /home/amicro/project/yolov3 && rm grpc-tensorrt.log && touch grpc-tensorrt.log
cd /home/amicro/project/yolov3 && nohup /usr/bin/python trt_grpc_detect_server.py --choose-config yolov5s_xavier > grpc-tensorrt.log 2>&1 &
