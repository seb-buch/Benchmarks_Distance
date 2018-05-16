import numpy as np
from MDAnalysis.lib.distances import _box_check

def init_uniform(box,Npoints):
	boxtype = _box_check(box)
	if boxtype == 'ortho': 
		box1 = box[:3] if box.shape == (6,) \
		else np.array((box[0][0],box[1][1],box[2][2]))
	points = np.random.uniform(low=0,high=1.0,size=(Npoints,3))*box1[:]
	points = np.array(points,dtype=np.float32)
	return box1,points