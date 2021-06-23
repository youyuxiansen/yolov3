import cv2
import os
import numpy as np


img = cv2.imread('/Users/felix/Documents/python_project/yolov3/runs/detect/tmp/nparr.jpg')
img_str = np.array(img).tobytes()
img_list = [b for b in img_str]
with open("/Users/felix/Downloads/nparr_bytes.txt", "w") as text_file:
	text_file.write(str(img_list))


