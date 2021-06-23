import argparse
from utils.inference.server.grpc_server import serve
import yaml


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--port', type=int, default=5000, help='specify port')
	parser.add_argument('--yaml-path', type=str, default='config/trt_config.yaml', help='yaml filepath of tensorrt inference config')
	opt = parser.parse_args()

	with open(opt.yaml_path) as f:
		data_dict = yaml.load(f, Loader=yaml.SafeLoader)
		opt.data_dict = data_dict.get('yolov5s')
	serve(opt)

