import argparse

# import asyncio
# import multiprocessing
import socket
import contextlib
# import sys
import numpy as np
import cv2
import sys

from utils.inference.server.grpc_server import serve, _PROCESS_COUNT
import yaml
from utils.inference.Processor import Processor
from utils.inference.Visualizer import Visualizer
from utils.general import scale_coords, xyxy2xywh
from utils.datasets import process_img


@contextlib.contextmanager
def _reserve_port():
    """Find and reserve a port for all subprocesses to use."""
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    if sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) == 0:
        raise RuntimeError("Failed to set SO_REUSEPORT.")
    sock.bind(('', 0))
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000, help='specify port')
    parser.add_argument('--yaml-path', type=str, default='config/trt_config.yaml',
                        help='Yaml filepath of tensorrt inference config')
    parser.add_argument('--choose-config', type=str, default='yolov5s', help='Choose tensorrt inference config in yaml')
    parser.add_argument('--test-infer-config', action='store_true', help='Test infer config in trt_config.yaml')
    opt = parser.parse_args()

    with open(opt.yaml_path) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
        opt.data_dict = data_dict.get(opt.choose_config)
    workers = []
    # loops = []
    processor = Processor(opt.data_dict.get('weights'),
                          opt.data_dict.get('anchor_nums'),
                          opt.data_dict.get('nc'),
                          np.array(opt.data_dict.get('anchors')),
                          np.array(opt.data_dict.get('output_shapes')),
                          opt.data_dict.get('imgsz'),
                          np.array(opt.data_dict.get('strides')))
    workers = []
    # multiprocessing.set_start_method('spawn')
    if opt.test_infer_config:
        img0 = cv2.imread("data/images/5948.jpg")
        img = process_img(img0, opt.data_dict.get('imgsz'))
        pred = processor.detect(img)
        bbox = []
        for _, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                xyxy = det[:, :4].astype(int).flatten().tolist()
                bbox.append(xyxy)
        visualizer = Visualizer()
        visualizer.plot_one_box(img0, np.array(bbox))
        cv2.imwrite("data/images/test_infer.jpg", img0)
        sys.exit()

    serve(opt, processor)
# for _ in range(1):
# 	# NOTE: It is imperative that the worker subprocesses be forked before
# 	# any gRPC servers start up. See
# 	# https://github.com/grpc/grpc/issues/16001 for more details.
# 	# loop = asyncio.get_event_loop()
# 	# loop.run_until_complete()
#
# 	worker = multiprocessing.Process(target=serve, args=(opt, processor, ))
# 	worker.start()
# 	workers.append(worker)
# 	# loops.append(loop)
# for worker in workers:
# 	worker.join()
# 	# loop.close()
