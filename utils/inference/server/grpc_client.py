# python版本的客户端
# client.py文件，同样的把那两个生成.py文件放到同一个目录下哦
import grpc
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import uploadPic_pb2
import uploadPic_pb2_grpc
import cv2
import base64
import numpy as np


def run():

	start = time.time()
	# 获取一个频道，呃，可能叫申请一个通讯线路合适点吧
	channel = grpc.insecure_channel('192.168.0.175:5000')
	# 获取远程函数的操作指针（Python没有得指针，不晓得怎么称呼这玩意）
	stub = uploadPic_pb2_grpc.uploadPicServicerStub(channel)
	# 打开图片
	img = cv2.imread('/home/yousixia/project/yolov3/data/images/5948.jpg')
	img_encode = cv2.imencode('.jpg', img)[1]
	# imgg = cv2.imencode('.png', img)
	data_encode = np.array(img_encode)
	str_encode = data_encode.tostring()
	img_64 = base64.b64encode(str_encode)

	# 调用服务
	response = stub.Upload(uploadPic_pb2.MatImage(
		mat_data=img_64, cols=1080, rows=1920, elt_size=3))
	# response.values就是返回值
	bbox = response.Bbox
	print('request time:', time.time() - start)


if __name__ == '__main__':
	while True:
		run()
