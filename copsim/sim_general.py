import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))
sys.path.append("/home/yuth/coppelia/programming/zmqRemoteApi/clients/python")
from zmqRemoteApi import RemoteAPIClient
import numpy as np

client = RemoteAPIClient()
sim = client.getObject('sim')

# Dummy
# hd = sim.createDummy(1)
# sim.setObjectPosition(hd,-1,[1,1,1])
# handles = [sim.createDummy(0.01, 12 * [0]) for _ in range(50)]
# for i, h in enumerate(handles):
#     sim.setObjectPosition(h, -1, [0.01 * i, 0.01 * i, 0.01 * i])
#     sim.setObjectOrientation(h,-1,[1.5,0,0]) # euler angle

# Point Cloud
pointCloudHandles = sim.getObject("./PointCloud")
# for i in range(700):
#     point = np.random.uniform(-2, 2, size=3)
#     point /= np.linalg.norm(point)
#     sim.insertPointsIntoPointCloud(pointCloudHandles, 
#                                    0, 
#                                    [point[0], point[1], point[2]], 
#                                    [155, 244, 100])

point = np.load("map/mapdata/point_cloud/xyz(0.9).npy")
color = np.load("map/mapdata/point_cloud/rgb(0.9).npy")
for i in range(point.shape[0]):
    print(i)
    sim.insertPointsIntoPointCloud(pointCloudHandles, 
                                   0, 
                                   [point[i,0], point[i,1], point[i,2]], 
                                   [int(color[i,0]), int(color[i,1]), int(color[i,2])])