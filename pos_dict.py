import pickle
import numpy as np
import os

with open("camera1_pos.pkl","rb") as f:
	camera1_pos = pickle.load(f)

with open("camera2_pos.pkl","rb") as f:
	camera2_pos = pickle.load(f)

with open("camera3_pos.pkl","rb") as f:
	camera3_pos = pickle.load(f)

with open("camera4_pos.pkl","rb") as f:
	camera4_pos = pickle.load(f)

with open("camera5_pos.pkl","rb") as f:
	camera5_pos = pickle.load(f)

pos_dict = {}

for key,value in camera1_pos.items():
	pos_dict[key] = value

for key,value in camera2_pos.items():
	pos_dict[key] = value

for key,value in camera3_pos.items():
	pos_dict[key] = value

for key,value in camera4_pos.items():
	pos_dict[key] = value

for key,value in camera5_pos.items():
	pos_dict[key] = value


file = open("pos_dict.pkl","wb")
pickle.dump(pos_dict,file)
file.close()
