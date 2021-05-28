import random

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np

ia.seed(1)

image = ia.quokka(size=(256, 256))
bbs = BoundingBoxesOnImage([
	BoundingBox(x1=65, y1=100, x2=200, y2=150),
	BoundingBox(x1=150, y1=80, x2=200, y2=130)
], shape=image.shape)

seq = iaa.Sequential([
	# iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
	iaa.Affine(
		# translate_px={"x": 40, "y": 60},  # 平移
		scale=2,  # 缩放(0.5, 1.5)
		# rotate=(0, 360)  # 旋转
	)  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
])

# Augment BBs and images.
image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

# print coordinates before/after augmentation (see below)
# use .x1_int, .y_int, ... to get integer coordinates
for i in range(len(bbs.bounding_boxes)):
	before = bbs.bounding_boxes[i]
	after = bbs_aug.bounding_boxes[i]
	print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
		i,
		before.x1, before.y1, before.x2, before.y2,
		after.x1, after.y1, after.x2, after.y2)
	      )

# image with BBs before/after augmentation (shown below)
image_before = bbs.draw_on_image(image, size=2)
image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])

ia.imshow(image_before)
ia.imshow(image_after)




GREEN = [0, 255, 0]
ORANGE = [255, 140, 0]
RED = [255, 0, 0]

# Pad image with a 1px white and (BY-1)px black border
def pad(image, by):
	image_border1 = ia.pad(image, top=1, right=1, bottom=1, left=1,
	                       mode="constant", cval=255)
	image_border2 = ia.pad(image_border1, top=by-1, right=by-1,
	                       bottom=by-1, left=by-1,
	                       mode="constant", cval=0)
	return image_border2

# Draw BBs on an image
# and before doing that, extend the image plane by BORDER pixels.
# Mark BBs inside the image plane with green color, those partially inside
# with orange and those fully outside with red.
def draw_bbs(image, bbs, border):
	image_border = pad(image, border)
	for bb in bbs.bounding_boxes:
		if bb.is_fully_within_image(image.shape):
			color = GREEN
		elif bb.is_partly_within_image(image.shape):
			color = ORANGE
		else:
			color = RED
		image_border = bb.shift(left=border, top=border) \
			.draw_on_image(image_border, size=2, color=color)

	return image_border

# Define example image with three small square BBs next to each other.
# Augment these BBs by shifting them to the right.
image = ia.quokka(size=(256, 256))
bbs = BoundingBoxesOnImage([
	BoundingBox(x1=25, x2=75, y1=25, y2=75),
	BoundingBox(x1=100, x2=150, y1=25, y2=75),
	BoundingBox(x1=175, x2=225, y1=25, y2=75)
], shape=image.shape)

seq = iaa.Affine(translate_px={"x": 120})
image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

# Draw the BBs (a) in their original form, (b) after augmentation,
# (c) after augmentation and removing those fully outside the image,
# (d) after augmentation and removing those fully outside the image and
# clipping those partially inside the image so that they are fully inside.
image_before = draw_bbs(image, bbs, 100)
image_after1 = draw_bbs(image_aug, bbs_aug, 100)
image_after2 = draw_bbs(image_aug, bbs_aug.remove_out_of_image(), 100)
image_after3 = draw_bbs(image_aug, bbs_aug.remove_out_of_image().clip_out_of_image(), 100)
ia.imshow(image_after2)
[x.compute_out_of_image_fraction(image) for x in bbs_aug]
ia.imshow(image_after3)


# compute iou
bb1 = BoundingBox(x1=50, x2=100, y1=25, y2=75)
bb2 = BoundingBox(x1=75, x2=125, y1=50, y2=100)

# Compute intersection, union and IoU value
# Intersection and union are both bounding boxes. They are here
# decreased/increased in size purely for better visualization.
bb_inters = bb1.intersection(bb2).extend(all_sides=-1)
bb_union = bb1.union(bb2).extend(all_sides=2)
iou = bb1.iou(bb2)