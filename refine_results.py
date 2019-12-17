import pickle
import numpy as np
import os

src_dir = "D:\\ml_related_codes\\speed_crops\\"
images = os.listdir(src_dir)

speed = {}

for img in images:
	ind = img.find('f') -1
	if img[:ind] not in speed:
		speed[img[:ind]] = 1
	else:
		speed[img[:ind]] += 1
cnt = 0
for img in images:
	ind = img.find('f') -1
	if  speed[img[:ind]] < 4:
		#cnt += 1
		os.remove(src_dir+img)

