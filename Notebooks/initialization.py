import numpy as np
from MDAnalysis.lib.distances import _box_check

def init_uniform(box,Npoints,constrain=None):
	points = np.random.uniform(low=0,high=1.0,size=(Npoints,3))*box[:3]
	points = np.array(points,dtype=np.float32)
	return points