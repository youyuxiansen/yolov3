# coding=utf-8
from concurrent import futures
import argparse
import time

import grpc
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from . import uploadPic_pb2
from . import uploadPic_pb2_grpc

import cv2
import numpy as np
import torch

from utils.inference import Processor
from utils.general import scale_coords, xyxy2xywh
from utils.datasets import process_img
# 设置最大传输字节数
MAX_MESSAGE_LENGTH = 2147483647
_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class UploadPicServicer(uploadPic_pb2_grpc.uploadPicServicerServicer):
	def __init__(self, processor, imgsz):
		self.processor = processor
		self.imgsz = imgsz

	# 工作函数
	def Upload(self, request, context):
		img0 = np.frombuffer(request.mat_data, dtype=np.uint8).reshape(request.rows, request.cols, request.channels)  # h, w, c
		cv2.imwrite('/home/yousixia/project/yolov3/runs/detect/tmp/nparr.jpg', img0)
		img = process_img(img0, self.imgsz)
		pred = self.processor.detect(img)
		bbox = []  # xywh
		for _, det in enumerate(pred):  # detections per image
			if len(det):
				# Rescale boxes from img_size to im0 size
				det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
				xywh = xyxy2xywh(det[:, :4]).flatten().tolist()
				xyxy = [int(x) for x in xywh]
				bbox += xyxy
		reply = uploadPic_pb2.Reply()
		print(bbox)
		reply.Bbox.extend(bbox)
		return reply


def serve(opt):
	processor = Processor(opt.data_dict.get('weights'),
	                      opt.data_dict.get('anchor_nums'),
	                      opt.data_dict.get('nc'),
	                      np.array(opt.data_dict.get('anchors')),
	                      np.array(opt.data_dict.get('output_shapes')),
	                      opt.data_dict.get('imgsz'))
	# gRPC 服务器
	server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
	                     options=[
		                     ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
		                     ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)
	                     ])
	uploadPic_pb2_grpc.add_uploadPicServicerServicer_to_server(UploadPicServicer(
		processor,
		opt.data_dict.get('imgsz')
	), server)
	server.add_insecure_port('[::]:' + str(opt.port))
	print("Server is opening ,waiting for message...")
	server.start()  # start() 不会阻塞，如果运行时你的代码没有其它的事情可做，你可能需要循环等待。
	server.wait_for_termination()



