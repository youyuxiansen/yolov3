import argparse
import time
from pathlib import Path
import queue
import os

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import matplotlib.pyplot as plt

from models.experimental import attempt_load
from utils.datasets import LoadStreamsStrided, LoadImagesStrided, LoadStreamsTrt, LoadImagesTrt
from utils.general import check_img_size, check_requirements, \
	check_imshow, non_max_suppression, apply_classifier, \
	scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box, plot_center_point, plot_move_routes, \
	one_cover_two_with_mask
from utils.amicro_plot import four_point_transform
from utils.torch_utils import select_device, load_classifier, time_synchronized

from utils.inference import Processor
import matplotlib.pyplot as plt


def trt_detect(save_img=False):
	# yolov3-ssp with evolve
	# anchor_nums = 4
	# nc = 1
	# anchors = np.array([
	#     [[11, 10], [17, 9], [18, 16], [29, 16]],
	#     [[34, 28], [48, 24], [59, 33], [46, 64]],
	#     [[69, 45], [86, 59], [96, 80], [150, 106]]
	# ])
	# output_shapes = [
	#     (1, anchor_nums, 80, 80, nc + 5),
	#     (1, anchor_nums, 40, 40, nc + 5),
	#     (1, anchor_nums, 20, 20, nc + 5)
	# ]

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

	source, weights, view_img, save_txt, imgsz = \
		opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
	save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
	webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
		('rtsp://', 'rtmp://', 'http://', 'https://'))

	# Directories
	save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
	(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

	# Initialize
	set_logging()

	stride = int(strides.max())  # model stride
	print(f"Loading trt engine!")
	# imgsz = check_img_size(imgsz, s=stride)  # check img_size

	# Set Dataloader
	vid_path, vid_writer = None, None
	bird_transform, pts = False, None
	if opt.plot_move_routes:
		bird_transform = True
		# Coordinates of chessboard region needs to apply brid transform
		pts = np.array([(567, 28), (1458, 60), (1890, 639), (638, 1068)])
		center_point = queue.Queue()

	if webcam:
		view_img = check_imshow()
		cudnn.benchmark = True  # set True to speed up constant image size inference
		dataset = LoadStreamsTrt(source, img_size=imgsz, bird_transform=bird_transform, pts=pts)
	else:
		dataset = LoadImagesTrt(source, img_size=imgsz, bird_transform=bird_transform, pts=pts)

	# Get names and colors
	names = ['Robot']
	# colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
	colors = [[96, 171, 132]]

	# Run inference
	t0 = time.time()
	img_index = 0
	# Create an empty picture and draw the route on it, which is independent from the real picture.
	route_mask = None
	cover_heatmap_accum = None
	processor = Processor(weights[0], anchor_nums, nc, anchors, output_shapes, imgsz)
	for path, img, im0s, vid_cap in dataset:
		if opt.plot_move_routes and route_mask is None:
			# initialize something for ploting move routes
			route_img = im0s.copy()
			route_mask = np.zeros((im0s.shape[0], im0s.shape[1], 3), np.uint8)
			route_mask.fill(255)
			route_mask_bg_color = 'white'
		if opt.cover_heatmap:
			if cover_heatmap_accum is None:
				# initialize something for ploting cover heatmap
				cover_heatmap_img = im0s.copy()
				# The template with a black base color is used to
				# continuously accumulate the covered elliptical areas going in.
				cover_heatmap_accum = np.zeros((im0s.shape[0], im0s.shape[1], 3), np.uint8)
				cover_heatmap_accum.fill(0)
				cover_heatmap_accum_bg_color = 'black'
				# The same img as img0s used to host cumulative heatmaps.
				cover_heatmap_accum_img0s = im0s.copy()
			# restore cover_heatmap_tmp every loop
			cover_heatmap_tmp = np.zeros((im0s.shape[0], im0s.shape[1], 3), np.uint8)
			cover_heatmap_tmp.fill(0)
		# Inference
		# t1 = time_synchronized()
		pred = processor.detect(img)

		# Process detections
		for i, det in enumerate(pred):  # detections per image
			if webcam:  # batch_size >= 1
				p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
			else:
				p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

			p = Path(p)  # to Path
			save_path = str(save_dir / p.name)  # img.jpg
			txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
			s += '%gx%g ' % img.shape[2:]  # print string
			gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
			if len(det):
				# Rescale boxes from img_size to im0 size
				det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
				det[:, 6:8] = scale_coords(img.shape[2:], det[:, 6:8], im0.shape).round()

				# plot move routes
				if opt.plot_move_routes and img_index % opt.move_routes_interval == 0:  # 设置绘制间隔
					assert pred[0].shape[0] == 1, "Only one robot can exist in the scene when drawing movement routes!"
					center_point.put(list(map(int, det[:, 6:8].tolist()[0])))
					# elements of center_point always Less than or equal to 2
					if center_point.qsize() <= 1:
						pass
					else:
						pts1 = center_point.get()
						pts2 = center_point.get()
						plot_move_routes([pts1, pts2], route_mask, colors[int(cls)], 3)
						route_img = one_cover_two_with_mask(route_mask, route_img, bg_color=route_mask_bg_color)
						cv2.imwrite('/home/yousixia/project/yolov3/runs/detect/tmp/im0s.jpg', im0s)
						center_point.put(pts2)

				# plot cover heatmap
				if opt.cover_heatmap and img_index % opt.cover_heatmap_interval == 0:
					x1, y1, x2, y2 = list(map(int, det[:, :4].tolist()[0]))
					# Draws a white ellipse with a center at center_point of bbox,
					# two axes is (x2-x1, y2-y1), and a line width of 3.
					cv2.ellipse(cover_heatmap_tmp, list(map(int, det[:, 6:8].tolist()[0])),
					            (int((x2 - x1) * 0.9 / 2), int((y2 - y1) * 0.9 / 2)), 0, 0, 360, (32, 16, 16),
					            -1)  # 画椭圆
					cover_heatmap_accum = cv2.addWeighted(cover_heatmap_accum, 1, cover_heatmap_tmp, 1, 0)  # 累加覆盖面积
					cover_heatmap_accum_colormap = cv2.applyColorMap(cover_heatmap_accum, cv2.COLORMAP_JET)
					one_cover_two_with_mask(cover_heatmap_accum, cover_heatmap_img, cover_heatmap_accum_colormap,
					                        bg_color='black')
				# cv2.imwrite('/home/yousixia/project/yolov3/runs/detect/tmp/original_img.jpg',
				#             original_img)

				# Print results
				for c in np.unique(det[:, 5]):
					n = (det[:, 5] == c).sum()  # detections per class
					s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

				# Write results
				for *xyxy, conf, cls, center_x, center_y in reversed(det):
					if save_txt:  # Write to file
						xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
						line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
						with open(txt_path + '.txt', 'a') as f:
							f.write(('%g ' * len(line)).rstrip() % line + '\n')

					if save_img or view_img:  # Add bbox to image
						label = f'{names[int(cls)]} {conf:.2f}'
						plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
					# if opt.plot_move_routes:
					# 	plot_one_box(xyxy, route_img, label=label, color=colors[int(cls)], line_thickness=3)
					# if opt.cover_heatmap:
					# 	plot_one_box(xyxy, cover_heatmap_img, label=label, color=colors[int(cls)], line_thickness=3)
					# 	cv2.imwrite('/home/yousixia/project/yolov3/runs/detect/tmp/cover_heatmap_img.jpg',
					# 	            cover_heatmap_img)

			# plot_center_point((center_x, center_y), im0, color=colors[int(cls)], line_thickness=3)
			# Print time (inference + NMS)
			# print(f'{s}Done. ({t2 - t1:.3f}s)')

			# drawing cover rate plot
			cover_rate_plot_data = []
			if opt.cover_rate:
				heatmap_non_zero_pixels = cv2.countNonZero(cover_heatmap_accum)
				all_pixels = cv2.countNonZero(im0s)
				cover_heatmap_img
				cover_rate = 1.0 * 100 * heatmap_non_zero_pixels / all_pixels
				secend = img_index / vid_cap.get(cv2.CAP_PROP_FPS)
				cover_rate_plot_data.append([cover_rate, secend])
				l1 = plt.plot(secend, cover_rate, 'b--', label='覆盖率')
				plt.xlabel('时间/s')
				plt.ylabel('覆盖率/%')
				plt.legend()
				# plt.show()



			# stack bbox, movement routes, cover heatmap, cover rate to one matrix.
			# TODO


			# Stream results
			if view_img:
				cv2.imshow(str(p), im0)
				cv2.waitKey(1)  # 1 millisecond

			# Save results (image with detections)
			if save_img:
				# cv2.imwrite('/home/yousixia/project/yolov3/runs/detect/tmp/123.jpg', im0)
				if dataset.mode == 'image':
					cv2.imwrite(save_path, im0)
				else:  # 'video' or 'stream'
					if vid_path != save_path:  # new video
						vid_path = save_path
						if isinstance(vid_writer, cv2.VideoWriter):
							vid_writer.release()  # release previous video writer
						if vid_cap:  # video
							fps = vid_cap.get(cv2.CAP_PROP_FPS)
							if opt.plot_move_routes:
								w, h = im0.shape[1], im0.shape[0]
							else:
								w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
								h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
						else:  # stream
							fps, w, h = 30, im0.shape[1], im0.shape[0]
							save_path += '.mp4'
						vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
						if opt.plot_move_routes:
							filename = save_path.split('/')[-1]
							movement_routes_vid_writer = cv2.VideoWriter(
								os.path.join('/'.join(save_path.split('/')[:-1]), 'movement_routes_' + filename),
								cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
						if opt.cover_heatmap:
							cover_heatmap_vid_writer = cv2.VideoWriter(
								os.path.join('/'.join(save_path.split('/')[:-1]), 'cover_heatmap_' + filename),
								cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
					vid_writer.write(im0)
					movement_routes_vid_writer.write(route_img)
					cover_heatmap_vid_writer.write(cover_heatmap_img)

		img_index += 1
	if save_txt or save_img:
		s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
		print(f"Results saved to {save_dir}{s}")

	print(f'Done. ({time.time() - t0:.3f}s)')


def detect(save_img=False):
	source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
	save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
	webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
		('rtsp://', 'rtmp://', 'http://', 'https://'))

	# Directories
	save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
	(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

	# Initialize
	set_logging()
	device = select_device(opt.device)
	half = device.type != 'cpu'  # half precision only supported on CUDA

	# Load model
	model = attempt_load(weights, map_location=device)  # load FP32 model
	stride = int(model.stride.max())  # model stride
	imgsz = check_img_size(imgsz, s=stride)  # check img_size
	if half:
		model.half()  # to FP16

	# Second-stage classifier
	classify = False
	if classify:
		modelc = load_classifier(name='resnet101', n=2)  # initialize
		modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

	# Set Dataloader
	vid_path, vid_writer = None, None
	if webcam:
		view_img = check_imshow()
		cudnn.benchmark = True  # set True to speed up constant image size inference
		dataset = LoadStreamsStrided(source, img_size=imgsz, stride=stride)
	else:
		dataset = LoadImagesStrided(source, img_size=imgsz, stride=stride)

	# Get names and colors
	names = model.module.names if hasattr(model, 'module') else model.names
	colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

	# Run inference
	if device.type != 'cpu':
		model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
	t0 = time.time()
	for path, img, im0s, vid_cap in dataset:
		img = torch.from_numpy(img).to(device)
		img = img.half() if half else img.float()  # uint8 to fp16/32
		img /= 255.0  # 0 - 255 to 0.0 - 1.0
		if img.ndimension() == 3:
			img = img.unsqueeze(0)

		# Inference
		t1 = time_synchronized()
		pred = model(img, augment=opt.augment)[0]

		# Apply NMS
		pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
		t2 = time_synchronized()

		# Apply Classifier
		if classify:
			pred = apply_classifier(pred, modelc, img, im0s)

		# Process detections
		for i, det in enumerate(pred):  # detections per image
			if webcam:  # batch_size >= 1
				p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
			else:
				p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

			p = Path(p)  # to Path
			save_path = str(save_dir / p.name)  # img.jpg
			txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
			s += '%gx%g ' % img.shape[2:]  # print string
			gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
			if len(det):
				# Rescale boxes from img_size to im0 size
				det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

				# Print results
				for c in det[:, -1].unique():
					n = (det[:, -1] == c).sum()  # detections per class
					s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

				# Write results
				for *xyxy, conf, cls in reversed(det):
					if save_txt:  # Write to file
						xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
						line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
						with open(txt_path + '.txt', 'a') as f:
							f.write(('%g ' * len(line)).rstrip() % line + '\n')

					if save_img or view_img:  # Add bbox to image
						label = f'{names[int(cls)]} {conf:.2f}'
						plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

			# Print time (inference + NMS)
			print(f'{s}Done. ({t2 - t1:.3f}s)')

			# Stream results
			if view_img:
				cv2.imshow(str(p), im0)
				cv2.waitKey(1)  # 1 millisecond

			# Save results (image with detections)
			if save_img:
				if dataset.mode == 'image':
					cv2.imwrite(save_path, im0)
				else:  # 'video' or 'stream'
					if vid_path != save_path:  # new video
						vid_path = save_path
						if isinstance(vid_writer, cv2.VideoWriter):
							vid_writer.release()  # release previous video writer
						if vid_cap:  # video
							fps = vid_cap.get(cv2.CAP_PROP_FPS)
							w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
							h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
						else:  # stream
							fps, w, h = 30, im0.shape[1], im0.shape[0]
							save_path += '.mp4'
						vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
					vid_writer.write(im0)

	if save_txt or save_img:
		s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
		print(f"Results saved to {save_dir}{s}")

	print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights', nargs='+', type=str, default='yolov3.pt', help='model.pt path(s)')
	parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
	parser.add_argument('--img-size', nargs='+', type=int, default=640, help='inference size (pixels)')
	parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
	parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
	parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
	parser.add_argument('--view-img', action='store_true', help='display results')
	parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
	parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
	parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
	parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
	parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
	parser.add_argument('--augment', action='store_true', help='augmented inference')
	parser.add_argument('--update', action='store_true', help='update all models')
	parser.add_argument('--project', default='runs/detect', help='save results to project/name')
	parser.add_argument('--name', default='exp', help='save results to project/name')
	parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
	parser.add_argument('--with-trt', action='store_true',
	                    help='inference with tensorrt engine(Only work with tensorrt inference)')
	parser.add_argument('--plot-move-routes', action='store_true',
	                    help='drawing the robot movement routes(only surpport video or streams)')
	parser.add_argument('--move-routes-interval', type=int, default=1,
	                    help='frame interval of drawing the robot movement routes')
	parser.add_argument('--cover-heatmap', action='store_true',
	                    help='drawing the cover heatmap(only surpport video or streams)')
	parser.add_argument('--cover-heatmap-interval', type=int, default=60,
	                    help='frame interval of drawing the cover heatmap')
	parser.add_argument('--cover-rate', action='store_true',
	                    help='drawing the cover rate plot(only surpport video or streams)')
	# parser.add_argument('--apply-amicro-demand', action='store_true',
	#                     help='Amicro demands include drawing the robot movement routes, bird transform, cover heatmap and cover rate')
	# parser.add_argument('--stride', type=int, default=32, help='model stride(Only work with tensorrt inference)')
	opt = parser.parse_args()
	print(opt)

	if opt.with_trt:
		trt_detect()
	else:
		check_requirements(exclude=('pycocotools', 'thop'))
		with torch.no_grad():
			if opt.update:  # update all models (to fix SourceChangeWarning)
				for opt.weights in ['yolov3.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt']:
					detect()
					strip_optimizer(opt.weights)
			else:
				detect()
