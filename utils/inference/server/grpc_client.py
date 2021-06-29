# python版本的客户端
# client.py文件，同样的把那两个生成.py文件放到同一个目录下哦
import grpc
import os
import sys
import time
import threading

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import uploadPic_pb2
import uploadPic_pb2_grpc
import cv2
import base64
import numpy as np


def upload(stub, id, img_bytes):
	# start = time.time()
	# 打开图片

	video_id = 1
	id1 = id
	while id1 - id < 100:
		response = stub.Upload(uploadPic_pb2.MatImage(
			mat_data=img_bytes, cols=1920, rows=1080, channels=3,
			id=id, video_id=video_id))
		id1 += 1
	# bbox = response.Bbox
	# print('request time:', time.time() - start)


def getBbox(stub):
	for id in range(1, 301):
		response = stub.GetBbox(uploadPic_pb2.MatImage(
			id=id, video_id=1))
		print('Bbox of {}:{} is '.format(1, id), response.Bbox)

def run():
	channel = grpc.insecure_channel('192.168.0.175:5000')
	stub = uploadPic_pb2_grpc.uploadPicServicerStub(channel)

	img = cv2.imread('/home/yousixia/project/yolov3/data/images/5948.jpg')
	# img_encode = cv2.imencode('.jpg', img)[1]
	data_encode = np.array(img)
	str_encode = data_encode.tostring()

	lock1 = threading.Lock()
	lock2 = threading.Lock()
	lock3 = threading.Lock()
	thread1 = threading.Thread(target=upload, args=(stub, 1, str_encode), name='thread1')
	thread2 = threading.Thread(target=upload, args=(stub, 101, str_encode), name='thread2')
	thread3 = threading.Thread(target=upload, args=(stub, 201, str_encode), name='thread3')
	thread1.start()
	thread2.start()
	thread3.start()
	thread1.join()
	thread2.join()
	thread3.join()
	getBbox(stub)


if __name__ == '__main__':
	run()
