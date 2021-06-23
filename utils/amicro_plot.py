# Including plot demand from amicro.
import argparse

import cv2
import numpy as np


# 鼠标点击获取坐标
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		xy = "%d,%d" % (x, y)
		a.append(x)
		b.append(y)
		cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
		cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
		            1.0, (0, 0, 0), thickness=1)
		cv2.imshow("image", img)
		print(x, y)


def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype="float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect


def four_point_transform(img, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype="float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
	# return the warped image
	return warped


if __name__ == '__main__':
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", help="path to the image file")
	ap.add_argument("-c", "--coords",
	                help="comma seperated list of source points")
	args = vars(ap.parse_args())
	# load the image and grab the source coordinates (i.e. the list of
	# of (x, y) points)
	# NOTE: using the 'eval' function is bad form, but for this example
	# let's just roll with it -- in future posts I'll show you how to
	# automatically determine the coordinates without pre-supplying them
	img = cv2.imread(args["image"])
	rechoose_perspective_mat = False
	if rechoose_perspective_mat:
		a = []
		b = []
		cv2.namedWindow("image")
		cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
		cv2.imshow("image", img)
		cv2.waitKey(0)
		print(a[0], b[0])

	pts = np.array(eval(args["coords"]), dtype="float32")
	# apply the four point tranform to obtain a "birds eye view" of
	# the image
	warped = four_point_transform(img, pts)
	# show the original and warped images
	cv2.imshow("Original", img)
	cv2.imshow("Warped", warped)
	cv2.imwrite('/Users/felix/Documents/python_project/yolov3/data/images/5948_warped.jpg', warped)
	cv2.waitKey(0)
