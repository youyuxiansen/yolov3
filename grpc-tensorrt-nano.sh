#!/bin/sh
export OPENBLAS_CORETYPE=ARMV8
YOLOPATH=/home/amicro/project/yolov3
ONNX=robot_tracking_20210908.onnx
TRTFILE=best_nano.trt
cd $YOLOPATH && rm grpc-tensorrt.log && touch grpc-tensorrt.log
if ! [ -f "$TRTFILE" ]; then
	if ! [ -f "$ONNX" ]; then
		echo "Cannot find the onnx file!" 1>&2
		exit 1
  fi
	trtexec --onnx="$ONNX" --saveEngine="$TRTFILE" --workspace=100
fi
nohup /usr/bin/python trt_grpc_detect_server.py --choose-config yolov5s_nano_robot_tracking > grpc-tensorrt.log 2>&1 &