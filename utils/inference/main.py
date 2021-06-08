import argparse
import cv2
import numpy as np

from inference import Processor
from inference import Visualizer


# from utils.inference.Processor import Processor
# from utils.inference.Visualizer import Visualizer


def cli():
	desc = 'Run TensorRT yolov5 visualizer'
	parser = argparse.ArgumentParser(description=desc)
	parser.add_argument('-m', '--model', default='yolov5s-simple-32.trt', help='trt engine file located in ./models',
	                    required=False)
	parser.add_argument('-i', '--image', default='sample_720p.jpg', help='image file path', required=False)
	args = parser.parse_args()
	return {'model': args.model, 'image': args.image}


def main(anchor_nums, nc, anchors, output_shapes, img_size):
	# parse arguments
	args = cli()

	# setup processor and visualizer
	processor = Processor(model=args['model'],
                          anchor_nums=anchor_nums,
                          nc=nc, anchors=anchors,
                          output_shapes=output_shapes,
                          img_size=img_size)
	visualizer = Visualizer()

	# fetch input
	# while 1:
	print('image arg', args['image'])
	img = cv2.imread(args['image'])

	# inference
	output = processor.detect(img)
	img = cv2.resize(img, tuple(img_size))

	# object visualization
	# object_grids = processor.extract_object_grids(output)
	# visualizer.draw_object_grid(img, object_grids, 0.1)

	# class visualization
	# class_grids = processor.extract_class_grids(output)
	# visualizer.draw_class_grid(img, class_grids, 0.01)

	# bounding box visualization
	boxes = processor.extract_boxes(output)

	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	# visualizer.draw_boxes(img, boxes)
	visualizer.plot_one_box(img, boxes)

	# final results
	# boxes, confs, classes = processor.post_process(output)
	# visualizer.draw_results(img, boxes, confs, classes)


def trt_infer(model, img, anchor_nums, nc, anchors, output_shapes, img_size):

	# setup processor and visualizer
	processor = Processor(model=model,
	                      anchor_nums=anchor_nums,
	                      nc=nc, anchors=anchors,
	                      output_shapes=output_shapes,
	                      img_size=img_size)
	visualizer = Visualizer()

	# fetch input
	# while 1:
	# print('image arg', source)
	# img = cv2.imread(source)

	# inference
	output = processor.detect(img)
	return output


if __name__ == '__main__':
	# anchor_nums = 4
	# nc = 1
	# anchors = np.array([
	# 	[[11, 10], [17, 9], [18, 16], [29, 16]],
	# 	[[34, 28], [48, 24], [59, 33], [46, 64]],
	# 	[[69, 45], [86, 59], [96, 80], [150, 106]]
	# ])
	# output_shapes = [
	# 	(1, anchor_nums, 80, 80, nc + 5),
	# 	(1, anchor_nums, 40, 40, nc + 5),
	# 	(1, anchor_nums, 20, 20, nc + 5)
	# ]
	# yolov3-ssp with evolve
	nc = 1
	conf = {
		'yolov3-ssp': [
			4, 1,
			np.array([
				[[11, 10], [17, 9], [18, 16], [29, 16]],
				[[34, 28], [48, 24], [59, 33], [46, 64]],
				[[69, 45], [86, 59], [96, 80], [150, 106]]
			]),
			[
				(1, 4, 80, 80, nc + 5),
				(1, 4, 40, 40, nc + 5),
				(1, 4, 20, 20, nc + 5)
			]
		],
		'yolov5s': [
			3, 1,
			np.array([
				[[10, 13], [16, 30], [33, 23]],  # P3/8
				[[30, 61], [62, 45], [59, 119]],  # P4/16
				[[116, 90], [156, 198], [373, 326]]
			]),
			[
				(1, 3, 60, 80, nc + 5),
				(1, 3, 30, 40, nc + 5),
				(1, 3, 15, 20, nc + 5)
			]
		],
		'yolov5l': [
			4, 1,
			np.array([[[10, 10], [17, 9], [18, 15], [29, 16]],
			          [[34, 28], [48, 23], [58, 32], [45, 64]],
			          [[69, 45], [86, 59], [96, 79], [149, 105]]]),
			[
				(1, 4, 80, 80, nc + 5),
				(1, 4, 40, 40, nc + 5),
				(1, 4, 20, 20, nc + 5)
			]
		]
	}
	anchor_nums, nc, anchors, output_shapes = conf['yolov5s']
	img_size = [640, 480] # width, height宽高
	main(anchor_nums, nc, anchors, output_shapes, img_size)
