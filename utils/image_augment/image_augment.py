import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import random

#引入图像增强的方式
from albumentations import (
	VerticalFlip,
	HorizontalFlip,
	Flip,
	RandomRotate90,
	Rotate,
	ShiftScaleRotate,
	CenterCrop,
	OpticalDistortion,
	GridDistortion,
	ElasticTransform,
	ImageCompression,
	HueSaturationValue,
	RGBShift,
	RandomBrightnessContrast,
	RandomContrast,
	Blur,
	MotionBlur,
	MedianBlur,
	GaussNoise,
	CLAHE,
	ChannelShuffle,
	InvertImg,
	RandomGamma,
	ToGray,
	PadIfNeeded,
	Compose,
	BboxParams,
	CoarseDropout
)
BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)

#可视化的时候一定要注意bbox是pascal_voc 格式[x_min, y_min, x_max, y_max]还是coco格式[x_min, y_min, width, height]，然后根据需要进行修改
def visualize_bbox(img, bbox, class_id, class_idx_to_name, color=BOX_COLOR, thickness=20):
	x_min, y_min, x_max, y_max = bbox
	x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
	cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
	class_name = class_idx_to_name[class_id]
	((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 21)
	cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
	cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,TEXT_COLOR, lineType=cv2.LINE_AA)
	return img


def visualize(annotations, category_id_to_name):
	img = annotations['image'].copy()
	for idx, bbox in enumerate(annotations['bboxes']):
		img = visualize_bbox(img, bbox, annotations['category_id'][idx], category_id_to_name)
	plt.figure()
	plt.imshow(img)
	plt.show()

def get_aug(aug, min_area=0., min_visibility=0.):
	return Compose(aug, bbox_params=BboxParams(format='pascal_voc', min_area=min_area,
	                                           min_visibility=min_visibility, label_fields=['category_id'])) #这里的format也要根据bbox的格式进行修改
# image = download_image('http://images.cocodataset.org/train2017/000000386298.jpg')
image = cv2.imread('/Users/felix/Documents/data/labelme/000000002.jpg')
annotations = {'image': image, 'bboxes': [[589,218,801,336]], 'category_id': [1]}#注意这类有多个框的时候，catagory_id也要对应多个
category_id_to_name = {1: 'Robot'}
visualize(annotations, category_id_to_name)
aug = get_aug([Blur(p=1, blur_limit=9)])
augmented = aug(**annotations)
print(augmented['category_id'])
visualize(augmented, category_id_to_name)
# aug = get_aug([VerticalFlip(p=1)])
# augmented = aug(**annotations)
# visualize(augmented, category_id_to_name)

# aug = get_aug([HorizontalFlip(p=1)])
# augmented = aug(**annotations)
# visualize(augmented, category_id_to_name)

# aug = get_aug([CenterCrop(p=1, height=1000, width=2000)])
# augmented = aug(**annotations)
# visualize(augmented, category_id_to_name)

# aug = get_aug([RandomRotate90(p=1)])
# augmented = aug(**annotations)
# print(augmented['category_id'])
# visualize(augmented, category_id_to_name)

augmentation_configuration = {
	'random_brightness_contrast_limit': random.randrange(-0.7, 0.7, 0.3),
	'r_shift_limit': range(-150, 140, 40),
	'g_shift_limit': range(-150, 140, 40),
	'b_shift_limit': range(-150, 140, 40),
	'image_compression_limit': range(5, 46, 20),
	'motion_blur_limit': range(7, 26, 1),
	'gauss_noise_limit': range(10, 70, 10)
}
aug = get_aug([
	RandomBrightnessContrast(contrast_limit=augmentation_configuration.get(
		'random_brightness_contrast_limit'), p=0.5),
	RGBShift(r_shift_limit=augmentation_configuration.get('r_shift_limit'),
	         g_shift_limit=augmentation_configuration.get('g_shift_limit'),
	         b_shift_limit=augmentation_configuration.get('b_shift_limit'),
	         p=0.5),
	ImageCompression(quality_lower=augmentation_configuration.get(
		'image_compression_limit'), quality_upper=100, p=0.5),
	MotionBlur(blur_limit=augmentation_configuration.get('motion_blur_limit'), p=0.5),
	GaussNoise(var_limit=augmentation_configuration.get('gauss_noise_limit'), p=0.5),
	CoarseDropout(max_holes=18,
	              max_height=22,
	              max_width=22,
	              min_holes=8,
	              min_height=10,
	              min_width=10, p=0.5),



])

