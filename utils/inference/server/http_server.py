#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from http.server import HTTPServer, BaseHTTPRequestHandler
import codecs
import urllib
import torch
import json
import sys
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

import cv2
import numpy as np
import base64

from utils.inference import Processor
from utils.general import scale_coords
from utils.datasets import LoadImagesTrt
from datasets import process_img

# yolov5s
anchor_nums = 3
nc = 1
anchors = np.array([
	[[10, 13], [16, 30], [33, 23]],  # P3/8
	[[30, 61], [62, 45], [59, 119]],  # P4/16
	[[116, 90], [156, 198], [373, 326]]
])
strides = np.array([8., 16., 32.])
output_shapes = [
	(1, anchor_nums, 60, 80, nc + 5),
	(1, anchor_nums, 30, 40, nc + 5),
	(1, anchor_nums, 15, 20, nc + 5)
	# (1, anchor_nums*60*80, nc + 5),
	# (1, anchor_nums*30*40, nc + 5),
	# (1, anchor_nums*15*20, nc + 5)
]
weights = '/home/yousixia/project/yolov3/runs/train/exp14/weights/best_175.trt'
imgsz = [640, 480]


def detection(msg):
	img_decode_as = msg['file'].encode('ascii')
	img_decode = base64.b64decode(img_decode_as)
	img_np_ = np.frombuffer(img_decode, np.uint8)
	img = cv2.imdecode(img_np_, cv2.COLOR_RGB2BGR)
	original_shape = msg['original_shape']
	img0 = np.reshape(img, original_shape)
	img = process_img(img0, imgsz)
	processor = Processor(weights, anchor_nums, nc, anchors, output_shapes, imgsz)
	pred = processor.detect(img)
	# pred = img0
	for i, det in enumerate(pred):  # detections per image
		gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
		if len(det):
			# Rescale boxes from img_size to im0 size
			# det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
			pass

class TestHTTPHandle(BaseHTTPRequestHandler):
	def setup(self):
		self.request.settimeout(10)  # 设定超时时间10秒
		BaseHTTPRequestHandler.setup(self)

	def _set_response(self):
		self.send_response(200)  # 设定HTTP服务器请求状态200
		self.send_header('Content-type', 'application/json')  # 设定HTTP服务器请求内容格式
		self.end_headers()

	def do_GET(self):

		buf = '深度学习模型计算API服务'
		self.protocal_version = 'HTTP/1.1'

		self._set_response()

		buf = bytes(buf, encoding="utf-8")  # 编码转换
		self.wfile.write(buf)  # 写出返回内容

	def do_POST(self):
		'''
		处理通过POST方式传递过来并接收的输入数据并计算和返回结果
		'''
		path = self.path  # 获取请求的URL路径
		print(path)
		# 获取post提交的数据
		datas = self.rfile.read(int(self.headers['content-length']))
		req = json.loads(datas.decode())

		'''
		输入数据已经存储于变量“datas”里，这里可插入用于深度学习计算的处理代码等
		假设计算结果存储于变量“buf”里
		'''
		# TODO:Client need to send img array and it's original size
		result = detection(req)


		self._set_response()

		self.send_header('Content-type', 'application/json')
		self.wfile.write(json.dumps(result).encode('utf-8'))  # 向客户端写出返回结果

def start_server(ip, port):

	http_server = HTTPServer((ip, int(port)), TestHTTPHandle)

	print('服务器已开启')

	try:
		http_server.serve_forever()  # 设置一直监听并接收请求
	except KeyboardInterrupt:
		pass
	http_server.server_close()
	print('HTTP server closed')


if __name__ == '__main__':
	start_server('0.0.0.0', 20000)  # For IPv4 Network Only
