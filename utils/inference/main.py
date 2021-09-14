import sys
# sys.path.append("..")
from Visualizer import Visualizer
from Processor import Processor
import numpy as np
import cv2
import argparse
import os

sys.path.append("..")

from utils.datasets import process_img

# from utils.inference.Processor import Procesor
# from utils.inference.Visualizer import Visualizer


def main(anchor_nums, nc, anchors, output_shapes, img_size, model, image):
    # parse arguments

    # setup processor and visualizer
    processor = Processor(model=model,
                          anchor_nums=anchor_nums,
                          nc=nc,
                          anchors=anchors,
                          output_shapes=output_shapes,
                          img_size=img_size,
                          strides=[8., 16., 32.])
    # visualizer = Visualizer()

    # fetch input
    # while 1:
    print('image arg', image)
    img = cv2.imread(image)

    # inference

    img = process_img(img, img_size)
    output = processor.detect(img)
    # img = cv2.resize(img, tuple(img_size))

    # object visualization
    # object_grids = processor.extract_object_grids(output)
    # visualizer.draw_object_grid(img, object_grids, 0.1)

    # class visualization
    # class_grids = processor.extract_class_grids(output)
    # visualizer.draw_class_grid(img, class_grids, 0.01)

    # bounding box visualization
    # boxes = processor.extract_boxes(output)
    boxes = output[0][:, :4]

    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # visualizer.draw_boxes(img, boxes)
    visualizer.plot_one_box(img, boxes)
    cv2.imwrite("data/images/test_infer.jpg", img)

    # final results
    # boxes, confs, classes = processor.post_process(output)
    # visualizer.draw_results(img, boxes, confs, classes)


def trt_infer(model, img, anchor_nums, nc, anchors, output_shapes, img_size):
    # setup processor and visualizer
    processor = Processor(model=model,
                          anchor_nums=anchor_nums,
                          nc=nc,
                          anchors=anchors,
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
            4, nc,
            np.array([[[11, 10], [17, 9], [18, 16], [29, 16]],
                      [[34, 28], [48, 24], [59, 33], [46, 64]],
                      [[69, 45], [86, 59], [96, 80], [150, 106]]]),
            [(1, 4, 80, 80, nc + 5), (1, 4, 40, 40, nc + 5),
             (1, 4, 20, 20, nc + 5)]
        ],
        'yolov5s': [
            3,
            nc,
            np.array([
                [[10, 13], [16, 30], [33, 23]],  # P3/8
                [[30, 61], [62, 45], [59, 119]],  # P4/16
                [[116, 90], [156, 198], [373, 326]]
            ]),
            [(1, 3, 60, 80, nc + 5), (1, 3, 30, 40, nc + 5),
             (1, 3, 15, 20, nc + 5)]
        ],
        'yolov5l': [
            4, nc,
            np.array([[[10, 10], [17, 9], [18, 15], [29, 16]],
                      [[34, 28], [48, 23], [58, 32], [45, 64]],
                      [[69, 45], [86, 59], [96, 79], [149, 105]]]),
            [(1, 4, 80, 80, nc + 5), (1, 4, 40, 40, nc + 5),
             (1, 4, 20, 20, nc + 5)]
        ],
        'robot_tracking_yolov5s': [
            4,
            nc,
            np.array([
                [[13, 8], [17, 15], [26, 13], [30, 26]],  # P3/8
                [[46, 22], [58, 32], [48, 55], [46, 64]],  # P4/16
                [[69, 45], [86, 59], [97, 80], [162, 126]]
            ]),
            [(1, 4, 60, 80, nc + 5), (1, 4, 30, 40, nc + 5),
             (1, 4, 15, 20, nc + 5)]
        ],
    }
    anchor_nums, nc, anchors, output_shapes = conf['robot_tracking_yolov5s']
    # img_size = [480, 640]  # width, height宽高
    img_size = [640, 480]  # width, height宽高
    model = "best_nano.trt"
    image = "data/images/5948.jpg"
    main(anchor_nums, nc, anchors, output_shapes, img_size, model, image)
