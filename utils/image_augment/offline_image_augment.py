import random

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np


def process_augmentations(aug_seq, image, bbs, threshold):
	"""
	Applying data augmentation. Removing the bbox when it is outside the
	image or the fraction of bbox area outside of the image plane > threshold
	Parameters
	----------
	aug_seq: imgaug.augmenters.Sequential
	image: image numpy array (H,W,C)
	bbs: imgaug.augmentables.bbs,BoundingBoxesOnImage
	threshold: float
		Deleting the bbox when the fraction of bbox area outside of the image
		plane > threshold

	Returns image, bbs
	-------

	"""
	if isinstance(aug_seq, (iaa.Sequential, iaa.Augmenter)):
		image_aug, bbs_aug = aug_seq(image=image, bounding_boxes=bbs)
		bbs_fraction_outside = [x.compute_out_of_image_fraction(image) for x in bbs_aug]
		if_reserved = np.less(bbs_fraction_outside, threshold)
		bbs_aug = BoundingBoxesOnImage([bbs_aug[i] for i in range(len(if_reserved)) if if_reserved[i]], shape=image_aug.shape)
		bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()
		return image_aug, bbs_aug
	else:
		raise TypeError('Aug_seq must be an instance of iaa.Sequential or iaa.Augmenter')


def test_process_augmentations():
	image = ia.quokka(size=(256, 256))
	bbs = BoundingBoxesOnImage([
		BoundingBox(x1=25, x2=75, y1=25, y2=75),
		BoundingBox(x1=100, x2=150, y1=25, y2=75),
		BoundingBox(x1=175, x2=225, y1=25, y2=75)
	], shape=image.shape)
	seq = iaa.Affine(translate_px={"x": 120})
	# clipping those partially inside the image
	image_aug1, bbs_aug1 = process_augmentations(seq, image, bbs, 0.3)
	# removing those partially inside the image
	image_aug2, bbs_aug2 = process_augmentations(seq, image, bbs, 0.2)
	image_aug1 = bbs_aug1.draw_on_image(image_aug1, color=[0, 255, 0], size=2, alpha=0.75)
	ia.imshow(image_aug1)
	image_aug2 = bbs_aug2.draw_on_image(image_aug2, color=[0, 255, 0], size=2, alpha=0.75)
	ia.imshow(image_aug2)


# visualize
if __name__ == '__main__':
	test_process_augmentations()



