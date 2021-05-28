import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import shutil
from multiprocessing.pool import ThreadPool


classes = ["Robot"]
root = "/home/yousixia/data"
sets = ['train', 'val', 'test']

if os.path.exists(os.path.join(root, 'yolo_voc_amicro')):
	shutil.rmtree(os.path.join(root, 'yolo_voc_amicro'))
def convert(size, box):
	dw = 1. / (size[0])
	dh = 1. / (size[1])
	x = (box[0] + box[1]) / 2.0 - 1
	y = (box[2] + box[3]) / 2.0 - 1
	w = box[1] - box[0]
	h = box[3] - box[2]
	x = x * dw
	w = w * dw
	y = y * dh
	h = h * dh
	return x, y, w, h


def convert_annotation(root_path, image_id):
	try:
		file = os.path.join(root_path, 'VOC2007_amicro/Annotations/%s.xml') % image_id
		in_file = open(file)
	except FileNotFoundError:
		print('file %s does not exist' % file)
		return None
	out_file = open(os.path.join(root_path, 'yolo_voc_amicro/VOC/labels/%s.txt') % image_id, 'w')
	tree = ET.parse(in_file)
	root = tree.getroot()
	size = root.find('size')
	w = int(size.find('width').text)
	h = int(size.find('height').text)

	for obj in root.iter('object'):
		difficult = obj.find('difficult').text
		cls = obj.find('name').text
		if cls not in classes or int(difficult) == 1:
			continue
		cls_id = classes.index(cls)
		xmlbox = obj.find('bndbox')
		b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
		     float(xmlbox.find('ymax').text))
		bb = convert((w, h), b)
		out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


# 从VOC annotations里取出框的数据，转换成yolo所用的txt格式
if not os.path.exists(os.path.join(root, 'yolo_voc_amicro/VOC/labels/')):
	os.makedirs(os.path.join(root, 'yolo_voc_amicro/VOC/labels/'))
for image_set in sets:
	image_ids = open(os.path.join(root, 'VOC2007_amicro/ImageSets/Main/%s.txt') % image_set).read().strip().split()
	list_file = open(os.path.join(root, 'yolo_voc_amicro/', '_%s.txt') % image_set, 'w')
	for image_id in image_ids:
		list_file.write(os.path.join(root, 'VOC2007_amicro/JPEGImages/%s.jpg\n') % image_id)
		if image_set != 'test':
			convert_annotation(root, image_id)
	list_file.close()


# 创建存放转换后train和val图片文件的目录
os.system('mkdir -p ' + os.path.join(root, 'yolo_voc_amicro/VOC/images'))

root = "/home/yousixia/data/"

def format_yolo_data(root, set_name):

	with open(os.path.join(root, 'yolo_voc_amicro/', '_{}.txt'.format(set_name)), 'r') as f:
		lines = f.readlines()
	os.system('mkdir ' + os.path.join(root, 'yolo_voc_amicro/VOC/images/{}'.format(set_name)))
	if set_name != 'test':
		os.system('mkdir ' + os.path.join(root, 'yolo_voc_amicro/VOC/labels/{}'.format(set_name)))
	for line in lines:
		line = "/".join(line.split('/')[-3:]).strip()
		line = os.path.join(root, line)
		if (os.path.exists(line)):
			# 图片文件移到images文件夹
			path = "cp " + line + " " + os.path.join(root, "yolo_voc_amicro/VOC/images/{}".format(set_name))
			# path = path.replace(' ', '\ ').replace('(','\(').replace(')','\)')
			os.system(path)
		if set_name != 'test':
			# 标签文件移到labels文件夹
			line = line.replace('JPEGImages', 'labels')
			line = line.replace('VOC2007_amicro', 'yolo_voc_amicro')
			line = line.replace('jpg', 'txt')
			line = "/".join(line.split('/')[-2:]).strip()
			line = os.path.join(root, 'yolo_voc_amicro/VOC', line)
			if (os.path.exists(line)):
				path = "cp " + line + " " + os.path.join(root, "yolo_voc_amicro/VOC/labels/{}".format(set_name))
				# path = path.replace(' ', '\ ').replace('(','\(').replace(')','\)')
				os.system(path)

# format_yolo_data(root, 'train')
# format_yolo_data(root, 'val')
# format_yolo_data(root, 'test')
results = ThreadPool(3).imap(lambda x: format_yolo_data(root, x),
                             ['train', 'val', 'test'])
for result in results:
	if result is not None:
		result.wait()
		result.close()


