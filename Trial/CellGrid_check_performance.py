import cellgrid
import MDAnalysis as mda
import numpy as np


def check_cellgrid(points,cellsize,box):
	capped_distance = cellgrid.capped_self_distance_array(points, cellsize, box=box)
	return capped_distance

#Comparison with MDA.lib
def mda_distances(points,cellsize,box):
	check = mda.lib.distances.distance_array(points,points,box)
	return check
	
##Define a box of size 100 X 100 X 100,
## Distribute random points uniformly in the box
## Define the cell size to be used in cellgrid 
box = np.ones(3).astype(np.float32)*100
points = np.random.uniform(low=0,high=1.0,size=(1000,3))*box
points = np.array(points,order='F',dtype=np.float32)
cellsize = 10.0
capped_distance = check_cellgrid(points,cellsize,box)
check = mda_distances(points,cellsize,box)
mask = (capped_distance[1]<cellsize)
row, col = np.where(check < cellsize)
np.testing.assert_equal((len(row)-points.shape[0])/2, (capped_distance[1])[mask].size)
print("Tests Passed")
	
if __name__ == "__main__":
	import timeit	
	setup = "from __main__ import check_cellgrid,mda_distances,box,points,cellsize; import numpy"
	print timeit.timeit("check_cellgrid(points,cellsize,box)",setup=setup,number=5)
	print timeit.timeit("mda_distances(points,cellsize,box)",setup=setup,number=5)
