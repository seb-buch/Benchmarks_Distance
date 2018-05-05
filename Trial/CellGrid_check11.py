import cellgrid
import MDAnalysis as mda
import numpy as np

##Define a box of size 100 X 100 X 100,
## Distribute random points uniformly in the box
## Define the cell size to be used in cellgrid
box = np.ones(3).astype(np.float32)*100
points = np.random.uniform(low=0,high=1.0,size=(1000,3))*box
points = np.array(points,order='F',dtype=np.float32)
cellsize = 10.0

capped_distance = cellgrid.capped_self_distance_array(points, cellsize, box=box)
mask = (capped_distance[1]<cellsize)
#print (capped_distance[1])[mask].shape,(capped_distance[0])[mask].shape

#Comparison with MDA.lib
check = mda.lib.distances.distance_array(points,points,box)
row, col = np.where(check < cellsize)
print (len(row)-points.shape[0])/2, (capped_distance[1])[mask].shape
