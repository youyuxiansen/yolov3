"""
An example that uses TensorRT's Python api to make inferences.
"""
import argparse
import ctypes
import os
import shutil
import random
import threading
import time

import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import torch

CONF_THRESH = 0.5
IOU_THRESHOLD = 0.4
global num
num=0


def NMS(dets, score, thresh):
    #x1、y1、x2、y2、以及score赋值
    # （x1、y1）（x2、y2）为box的左上和右下角标
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = score
    #每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #order是按照score降序排序的，得到的是排序的本来的索引，不是排完序的原数组
    order = scores.argsort()[::-1]
    # ::-1表示逆序

    temp = []
    while order.size > 0:
        i = order[0]
        temp.append(i)
        #计算当前概率最大矩形框与其他矩形框的相交框的坐标
        # 由于numpy的broadcast机制，得到的是向量
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        #计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，需要用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #计算重叠度IoU
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        #找到重叠度不高于阈值的矩形框索引
        inds = np.where(ovr <= thresh)[0]
        #将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]
    return temp

def get_img_path_batches(batch_size, img_dir):
    ret = []
    batch = []
    for root, dirs, files in os.walk(img_dir):
        for name in files:
            if len(batch) == batch_size:
                ret.append(batch)
                batch = []
            batch.append(os.path.join(root, name))
    if len(batch) > 0:
        ret.append(batch)
    return ret

def get_img_path(img_dir):
    ret = []
    for root, dirs, files in os.walk(img_dir):
        for name in files:
            ret.append(os.path.join(root, name))
    return ret

def preprocess_image(img):
    """
    description: Read an image from image path, convert it to RGB,
                 resize and pad it to target size, normalize to [0,1],
                 transform to NCHW format.
    param:
        input_image_path: str, image path
    return:
        image:  the processed image
        image_raw: the original image
        h: original height
        w: original width
    """
    print('original image shape', img.shape)
    # img = cv2.resize(img, tuple(img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img.transpose((2, 0, 1)).astype(np.float16)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.0
    return img

def detect_preprocess_image(img):
    """
    description: Read an image from image path, convert it to RGB,
                 resize and pad it to target size, normalize to [0,1],
                 transform to NCHW format.
    param:
        input_image_path: str, image path
    return:
        image:  the processed image
        image_raw: the original image
        h: original height
        w: original width
    """
    # print('original image shape', img.shape)
    # img = cv2.resize(img, tuple(img_size))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img.transpose((2, 0, 1)).astype(np.float16)
    img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
    img = img / 255.0
    return img

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


class YoLov5TRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path, anchor_nums, nc, anchors, output_shapes, img_size):

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()
        # Create a Context on this device,
        self.cfx = cuda.Device(0).make_context()
        self.stream = cuda.Stream()

        inputs, outputs, bindings = [], [], []
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                inputs.append({'host': host_mem, 'device': cuda_mem})
            else:
                outputs.append({'host': host_mem, 'device': cuda_mem})

        # Store
        self.engine = engine
        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings
        self.anchor_nums = anchor_nums
        self.nc = nc  # classes
        self.no = self.nc + 5  # outputs per anchor
        # post processing config
        self.output_shapes = output_shapes
        self.strides = np.array([8., 16., 32.])
        self.na = len(anchors[0])
        self.nl = len(anchors)
        self.img_size = img_size
        a = anchors.copy().astype(np.float32)
        a = a.reshape(self.nl, -1, 2)
        self.anchors = a.copy()
        self.anchor_grid = a.copy().reshape(self.nl, 1, -1, 1, 1, 2)

    def inference(self, input_image):
        # threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.cfx.push()

        # Do image preprocess
        # batch_image_raw = []
        # batch_origin_h = []
        # batch_origin_w = []
        # batch_input_image = np.empty(shape=[self.batch_size, 3, self.input_h, self.input_w])
        # for i, image_raw in enumerate(raw_image_generator):
        #     input_image, image_raw, origin_h, origin_w = self.preprocess_image(image_raw)
        #     batch_image_raw.append(image_raw)
        #     batch_origin_h.append(origin_h)
        #     batch_origin_w.append(origin_w)
        #     np.copyto(batch_input_image[i], input_image)
        # input_image = np.ascontiguousarray(input_image)
        # Copy input image to host buffer
        self.inputs[0]['host'] = np.ravel(input_image)
        # Transfer input data  to the GPU.
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # Run inference.
        self.context.execute_async_v2(bindings=self.bindings,
                                      stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'],
                                   out['device'],
                                   self.stream)
        # Synchronize the stream
        self.stream.synchronize()
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()
        # Here we use the first row of output in that batch_size = 1
        outputs = [out['host'] for out in self.outputs]
        reshaped = []
        for output, shape in zip(outputs, self.output_shapes):
            reshaped.append(output.reshape(shape))
        # Do postprocess
        return reshaped

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()
        del self.ctx

    def get_raw_image(self, image_path):
        """
        description: Read an image from image path
        """
        yield cv2.imread(img_path)

    def get_raw_image_zeros(self):
        """
        description: Ready data for warmup
        """
        return np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def sigmoid_v(self, array):
        return np.reciprocal(np.exp(-array) + 1.0)

    def non_max_suppression(self, boxes, confs, classes, iou_thres=0.6):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = confs.flatten().argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_thres)[0]
            order = order[inds + 1]
        boxes = boxes[keep]
        confs = confs[keep]
        classes = classes[keep]
        return boxes, confs, classes

    def make_grid(self, nx, ny):
        """
		Create scaling tensor based on box location
		Source: https://github.com/ultralytics/yolov5/blob/master/models/yolo.py
		Arguments
			nx: x-axis num boxes
			ny: y-axis num boxes
		Returns
			grid: tensor of shape (1, 1, nx, ny, 80)
		"""
        nx_vec = np.arange(nx)
        ny_vec = np.arange(ny)
        yv, xv = np.meshgrid(ny_vec, nx_vec)
        grid = np.stack((yv, xv), axis=2)
        grid = grid.reshape(1, 1, ny, nx, 2)
        return grid

    def post_process(self, outputs, conf_thres):
        """
        description: postprocess the prediction
        param:
            outputs:     A tensor likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...]
            conf_thres:
        return:
            result_boxes: finally boxes, a boxes tensor, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a tensor, each element is the score correspoing to box
            result_classid: finally classid, a tensor, each element is the classid correspoing to box
        """
        # Get the num of boxes detected
        scaled = []
        grids = []
        for out in outputs:
            out = self.sigmoid_v(out)
            _, _, width, height, _ = out.shape
            grid = self.make_grid(width, height)
            grids.append(grid)
            scaled.append(out)
        z = []
        for out, grid, stride, anchor in zip(scaled, grids, self.strides, self.anchor_grid):
            _, _, width, height, _ = out.shape
            out[..., 0:2] = (out[..., 0:2] * 2. - 0.5 + grid) * stride
            out[..., 2:4] = (out[..., 2:4] * 2) ** 2 * anchor

            out = out.reshape((1, self.anchor_nums * width * height, self.no))
            z.append(out)
        pred = np.concatenate(z, 1)
        xc = pred[..., 4] > conf_thres
        pred = pred[xc]
        boxes = self.xywh2xyxy(pred[..., 0:4])
        # best class only
        confs = np.amax(pred[:, 5:], 1, keepdims=True)
        classes = np.argmax(pred[:, 5:], axis=-1)
        result_boxes, result_confs, result_classes = \
            self.non_max_suppression(boxes, confs, classes)
        return result_boxes, result_confs, result_classes

def open_cam_rtsp(cameraName,latancy,width,height):
    gst_str = ('rtspsrc location={} latency={} ! '
               'rtph264depay ! h264parse ! omxh264dec ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(cameraName, latancy, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


class inferThread(threading.Thread):
    def __init__(self, yolov5_wrapper, img_path):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = yolov5_wrapper
        self.img_path = img_path

    def run(self):
        img, use_time = self.yolov5_wrapper.infer_with_preprocess_postprocess(self.yolov5_wrapper.get_raw_image(self.img_path))
        parent, filename = os.path.split(img)
        save_name = os.path.join('output', filename)
        # Save image
        cv2.imwrite(save_name, img)
        print('input->{}, time->{:.2f}ms, saving into output/'.format(self.img_path, use_time * 1000))


class warmUpThread(threading.Thread):
    def __init__(self, yolov5_wrapper):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = yolov5_wrapper

    def run(self):
        batch_image_raw, use_time = self.yolov5_wrapper.infer_with_preprocess_postprocess(self.yolov5_wrapper.get_raw_image_zeros())
        print('warm_up->{}, time->{:.2f}ms'.format(batch_image_raw.shape, use_time * 1000))


if __name__ == "__main__":

    desc = 'Run TensorRT yolov5 visualizer'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-p', '--plugin', default='/home/amicro/project/tensorrtx/yolov5/build/libmyplugins.so', help='specify custom plugins library')
    parser.add_argument('-m', '--model', default='build/yolov5s.engine', help='trt engine file')
    parser.add_argument('-i', '--image', default='/home/amicro/data/yolo_voc_amicro/VOC/images/test', help='image file path')
    args = parser.parse_args()

    ctypes.CDLL(args.plugin)

    # load coco labels

    categories = ["Robot"]

    if os.path.exists('output/'):
        shutil.rmtree('output/')
    os.makedirs('output/')

    anchor_nums = 4
    nc = 1
    anchors = np.array([
        [[11, 10], [17, 9], [18, 16], [29, 16]],
        [[34, 28], [48, 24], [59, 33], [46, 64]],
        [[69, 45], [86, 59], [96, 80], [150, 106]]
    ])
    # a YoLov5TRT instance
    yolov5_wrapper = YoLov5TRT(args.model, anchor_nums, nc, anchors)
    try:
        # print('batch size is', yolov5_wrapper.batch_size)
        image_paths = get_img_path(args.image)
        for _ in range(2):
            # create a new thread to do warm_up
            thread1 = warmUpThread(yolov5_wrapper)
            thread1.start()
            thread1.join()
        for img in image_paths:
            # create a new thread to do inference
            thread1 = inferThread(yolov5_wrapper, img)
            thread1.start()
            thread1.join()
    finally:
        # destroy the instance
        yolov5_wrapper.destroy()

    # camera_addr='rtsp://admin:mkls1123@192.168.0.64/'
    # cap=open_cam_rtsp(camera_addr,200,1280,720)
    # #cap=cv2.VideoCapture("1.mp4")
    # # load custom plugins
    # PLUGIN_LIBRARY = "libmyplugins.so"
    # ctypes.CDLL(PLUGIN_LIBRARY)
    # engine_file_path = "yolov5s.engine"
    #
    # # load coco labels
    #
    # categories = ["person", "hat"]
    #
    # # a  YoLov5TRT instance
    # yolov5_wrapper = YoLov5TRT(engine_file_path)
    #
    # while True:
    #     ret,Frame=cap.read()
    #     if ret==True:
    #         time.sleep(0.00001)
    #         yolov5_wrapper.infer(Frame)
    #
    #     else:
    #         self.cap=open_cam_rtsp(camera_addr,200,1280,720)
    #         time.sleep(5)
    # # destroy the instance
    # yolov5_wrapper.destroy()



