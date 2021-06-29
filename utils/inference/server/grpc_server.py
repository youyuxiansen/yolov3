# coding=utf-8
from concurrent import futures
import argparse
import time
# from collections import OrderedDict
import threading

import grpc
import sys
import os

import logging

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
MAX_MESSAGE_LENGTH = 21474836
_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class UploadPicServicer(uploadPic_pb2_grpc.uploadPicServicerServicer):
	def __init__(self, processor, imgsz):
		self.processor = processor
		self.imgsz = imgsz
		self.prepared_img = dict()
		self.mutex = threading.Lock()

	# 工作函数
	def GetBbox(self, request, context):
		reply = uploadPic_pb2.Reply()
		self.mutex.acquire()
		try:
			item = self.prepared_img[request.video_id].pop(request.id)
		except KeyError:
			self.mutex.release()
			reply.message = 'Request img id {} of video id {} is not exists'.format(request.video_id, request.id)
			reply.request_state = False
			return reply
		self.mutex.release()
		img0 = np.frombuffer(item['img'], dtype=np.uint8).reshape(item['rows'], item['cols'], item['channels'])  # h, w, c
		# cv2.imwrite('/home/yousixia/project/yolov3/runs/detect/tmp/nparr.jpg', img0)
		img = process_img(img0, self.imgsz)
		pred = self.processor.detect(img)
		bbox = [request.video_id, request.id]  # return [video_id, id, xywh, xywh, ...]
		for _, det in enumerate(pred):  # detections per image
			if len(det):
				# Rescale boxes from img_size to im0 size
				det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
				xywh = xyxy2xywh(det[:, :4]).flatten().tolist()
				xyxy = [int(x) for x in xywh]
				bbox += xyxy
		print(bbox)
		reply.Bbox.extend(bbox)
		reply.request_state = True
		return reply

	def Upload(self, request, context):
		reply = uploadPic_pb2.Reply()
		self.insert(request)
		reply.request_state = True
		reply.message = 'Request img id {} of video id {} saved.'.format(request.video_id, request.id)
		return reply

	def insert(self, request):
		self.mutex.acquire()
		try:
			self.prepared_img[request.video_id]
		except KeyError:
			self.prepared_img[request.video_id] = dict()
		try:
			self.prepared_img[request.video_id][request.id]
		except KeyError:
			self.prepared_img[request.video_id][request.id] = dict()
		finally:
			self.prepared_img[request.video_id][request.id]['img'] = request.mat_data
			self.prepared_img[request.video_id][request.id]['channels'] = request.channels
			self.prepared_img[request.video_id][request.id]['rows'] = request.rows
			self.prepared_img[request.video_id][request.id]['cols'] = request.cols
			self.mutex.release()


def serve(opt):
	logging.basicConfig(level=logging.INFO)
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
	server.start()
	server.wait_for_termination()



