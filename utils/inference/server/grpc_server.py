# coding=utf-8
from concurrent import futures
# import argparse
import multiprocessing
import time
from datetime import datetime
# from collections import OrderedDict
import threading

import grpc
# from grpc.experimental import aio
import sys
import os

import logging

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from . import uploadPic_pb2
from . import uploadPic_pb2_grpc

# import cv2
import numpy as np
# import torch

# from utils.inference import Processor
from utils.general import scale_coords, xyxy2xywh
from utils.datasets import process_img
# 设置最大传输字节数
MAX_MESSAGE_LENGTH = 6300000
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_PROCESS_COUNT = multiprocessing.cpu_count()
_THREAD_CONCURRENCY = _PROCESS_COUNT


class UploadPicServicer(uploadPic_pb2_grpc.uploadPicServicerServicer):
	# @profile
	def __init__(self, processor, imgsz):
		self.processor = processor
		self.imgsz = imgsz
		self.bbox = dict()
		self.mutex = threading.Lock()

	# def GetBbox(self, request, context):
	# 	start = time.time()
	# 	reply = uploadPic_pb2.Reply()
	# 	self.mutex.acquire()
	# 	if request.video_id in self.bbox.keys() and request.id in self.bbox[request.video_id].keys():
	# 		bbox = self.bbox[request.video_id].pop(request.id)
	# 	else:
	# 		self.mutex.release()
	# 		reply.message = 'Request img id {} of video id {} is not exists'.format(request.video_id, request.id)
	# 		reply.request_state = False
	# 		return reply
	# 	self.mutex.release()
	# 	reply.Bbox.extend(bbox)
	# 	reply.request_state = True
	# 	print('getbbox time:', time.time() - start)
	# 	return reply

	def do_detect(self, img0, video_id, id):
		# cv2.imwrite('/home/yousixia/project/yolov3/runs/detect/tmp/nparr.jpg', img0)
		start = time.time()
		img = process_img(img0, self.imgsz)
		pred = self.processor.detect(img)
		print("finish a trt inference at ", datetime.utcnow().strftime('%Y/%m/%d %H:%M:%S.%f'))
		bbox = [video_id, id]  # return [video_id, id, xywh, xywh, ...]
		for _, det in enumerate(pred):  # detections per image
			if len(det):
				# Rescale boxes from img_size to im0 size
				det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
				xywh = xyxy2xywh(det[:, :4]).flatten().tolist()
				xyxy = [int(x) for x in xywh]
				bbox += xyxy
		print(bbox)
		print('do_detect time:', time.time() - start)
		return bbox

	# @profile
	def Upload(self, request, context):
		print("get image ", request.id)
		start = time.time()
		reply = uploadPic_pb2.Reply()
		img0 = np.frombuffer(request.mat_data, dtype=np.uint8).reshape(request.rows, request.cols,
		                                                               request.channels)  # h, w, c
		bbox = self.do_detect(img0, request.video_id, request.id)
		reply.Bbox.extend(bbox)
		reply.request_state = True
		# request_state, message = self.insert(request, bbox)
		# reply.request_state = request_state
		# reply.message = message
		print('Upload time:', time.time() - start)
		return reply

	def insert(self, request, bbox):
		self.mutex.acquire()
		if request.video_id not in self.bbox.keys():
			self.bbox[request.video_id] = dict()
		if request.id in self.bbox[request.video_id].keys():
			request_state = False
			message = 'Request img id {} of video id {} already exists.'.format(request.video_id, request.id)
		else:
			self.bbox[request.video_id][request.id] = bbox
			request_state = True
			message = 'Request img id {} of video id {} saved.'.format(request.video_id, request.id)
		self.mutex.release()
		return request_state, message


# @profile
def serve(opt, processor):
	logging.basicConfig(level=logging.INFO)
	# gRPC 服务器
	server = grpc.server(futures.ThreadPoolExecutor(max_workers=_THREAD_CONCURRENCY),
	                    options=[
		                    ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
		                    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)
	                    ])
	uploadPic_pb2_grpc.add_uploadPicServicerServicer_to_server(UploadPicServicer(
		processor,
		opt.data_dict.get('imgsz')
	), server)
	# server.add_insecure_port('[::]:' + str(opt.port))
	server.add_insecure_port('127.0.0.1:' + str(opt.port))
	print("Server is opening ,waiting for message...")
	server.start()
	try:
		server.wait_for_termination()
	except KeyboardInterrupt:
		server.stop(None)

