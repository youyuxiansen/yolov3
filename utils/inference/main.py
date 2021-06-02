import cv2
import sys
import argparse

from Processor import Processor
from Visualizer import Visualizer

def cli():
    desc = 'Run TensorRT yolov5 visualizer'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-m', '--model', default='yolov5s-simple-32.trt', help='trt engine file located in ./models', required=False)
    parser.add_argument('-i', '--image', default='sample_720p.jpg', help='image file path', required=False)
    args = parser.parse_args()
    return { 'model': args.model, 'image': args.image }

def main():
    # parse arguments
    args = cli()

    # setup processor and visualizer
    processor = Processor(model=args['model'])
    visualizer = Visualizer()

    # fetch input
    print('image arg', args['image'])
    img = cv2.imread(args['image'])

    # inference
    output = processor.detect(img) 
    img = cv2.resize(img, (640, 640))

    # object visualization
    # object_grids = processor.extract_object_grids(output)
    # visualizer.draw_object_grid(img, object_grids, 0.1)

    # class visualization
    # class_grids = processor.extract_class_grids(output)
    # visualizer.draw_class_grid(img, class_grids, 0.01)

    # bounding box visualization
    boxes = processor.extract_boxes(output)
    # visualizer.draw_boxes(img, boxes)
    visualizer.plot_one_box(img, boxes)

    # final results
    boxes, confs, classes = processor.post_process(output)
    visualizer.draw_results(img, boxes, confs, classes)


def trt_infer(model):
    # return trt infer method
    processor = Processor(model)
    return processor.detect


if __name__ == '__main__':
    main()   
