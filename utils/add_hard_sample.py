import os
from glob import glob

hard_samples_with_bbox = os.listdir('/Users/felix/Documents/data/labelme_difficult_with_bbox/')
for sample in hard_samples_with_bbox:
	os.system("cp " + os.path.join('/Users/felix/Documents/data/labelme/', sample)\
	          + " /Users/felix/Documents/data/labelme_difficult/")



