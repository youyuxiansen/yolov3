import os
from glob import glob
import shutil

from sklearn.model_selection import train_test_split

# 1.标签路径
data_root = '/home/yousixia/data/images'  # 图片的根目录
labelme_path = [
    os.path.join(data_root, "labelme/"),
    os.path.join(data_root, "labelme_difficult202105141453/"),
    os.path.join(data_root, "labelme_20210520/"),
    os.path.join(data_root, "negative_image/"),
]  # 原始labelme标注数据路径， 文件夹里包含jpg和xml文件，negative_image文件夹仅包含jpg，放加强训练用的负样本。
saved_path = os.path.join(data_root, "VOC2007_amicro/")  # 保存路径

# 2.创建要求文件夹
if not os.path.exists(saved_path + "Annotations"):
    os.makedirs(saved_path + "Annotations")
if not os.path.exists(saved_path + "JPEGImages/"):
    os.makedirs(saved_path + "JPEGImages/")
if not os.path.exists(saved_path + "ImageSets/Main/"):
    os.makedirs(saved_path + "ImageSets/Main/")

xml_files = []
image_files = []
# 3.复制xml文件到 VOC2007_amicro/Annotations/下
# 3.复制xml对应的jpg到 VOC2007_amicro/JPEGImages/下
for path in labelme_path:
    xml_files = glob(path + "*.xml")
    image_files = glob(path + "*.jpg")
    for xml_file in xml_files:
        shutil.copy(xml_file, saved_path + "Annotations/")
    for image in image_files:
        shutil.copy(image, saved_path + "JPEGImages/")

print("Finished copy image files to VOC2007_amicro/Annotations/")
print("Finished copy image files to VOC2007_amicro/JPEGImages/")

# 5.split files for txt
txtsavepath = saved_path + "ImageSets/Main/"
ftrainval = open(txtsavepath + '/trainval.txt', 'w')
ftest = open(txtsavepath + '/test.txt', 'w')
ftrain = open(txtsavepath + '/train.txt', 'w')
fval = open(txtsavepath + '/val.txt', 'w')
trainval_files = glob(os.path.join(saved_path, "Annotations/*.xml"))
trainval_files = [i.split("/")[-1].split(".xml")[0] for i in trainval_files]
total_files = os.listdir(os.path.join(data_root, "VOC2007_amicro/JPEGImages"))
test_files = [x for x in total_files if x not in trainval_files]
test_files = [i.split("/")[-1].split(".jpg")[0] for i in test_files]
for file in trainval_files:
    ftrainval.write(file + "\n")
# test
for file in test_files:
   ftest.write(file + "\n")
# split
train_files, val_files = train_test_split(trainval_files, test_size=0.15, random_state=42)
# train
# Add some negative image samples to train
negative_images = os.listdir(os.path.join(data_root, 'negative_image/'))
if not negative_images:
    train_files += [x.split(".jpg")[0] for x in negative_images]
for file in train_files:
    ftrain.write(file + "\n")

# val
for file in val_files:
    fval.write(file + "\n")

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()