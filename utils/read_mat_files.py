import os
import scipy.io as scio

path = '/home/yousixia/data/ObjectNet3D/Annotations'
for filename in os.listdir(path):
	data = scio.loadmat(os.path.join(path, filename))



