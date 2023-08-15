import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))
import numpy as np
from planner_dev.copsim_rrt_component import Node
from planner_util.extract_path_class import extract_path_class_6d
from copsim.arm_api import UR5eStateArmCoppeliaSimAPI, UR5eVirtualArmCoppeliaSimAPI
import time 

"""Number of Node : 313
Total Planning Time : 407.75942292
KCD Time Spend : 406.55507614500004
Planning Time Only : 1.2043467749999763
Number of Collision Check : 19001
Average KCD Time : 0.0213965094545024
"""
pathArray = np.array([[-0.4831071, 0.0514872, -0.4639085, -0.2666863, 1.5259414, 0.0000000], [-0.5586294, -0.1750165, -0.5203916, -0.2223494, 1.5765563, 0.1589794],
                      [-0.6527066, -0.1316107, -0.5903881, -0.0727589, 1.5125413, 0.3778187], [-0.6191012, -0.1732084, -0.6953801, 0.0770846, 1.3058748, 0.4824749],
                      [-0.6525155, -0.0043750, -0.7590095, 0.0256255, 1.1035960, 0.5954587], [-0.6220110, -0.1319450, -0.9280019, 0.0334071, 0.9004922, 0.6495439],
                      [-0.5650459, -0.0516826, -1.0775402, 0.0241605, 0.7200794, 0.8086570], [-0.6423592, -0.1272435, -0.9640069, 0.1936160, 0.5341623, 0.8549496],
                      [-0.6757667, -0.0488557, -1.0008803, 0.4666374, 0.4944516, 0.7824097], [-0.8289721, -0.0701928, -1.1891779, 0.5576380, 0.6394247, 0.8187217],
                      [-0.9332841, -0.1196766, -1.2990830, 0.6762730, 0.4819240, 0.6583768], [-0.9039973, -0.1212997, -1.4395539, 0.8621484, 0.3344253, 0.5439127],
                      [-0.7413957, -0.2636892, -1.5147516, 1.0121855, 0.2372141, 0.6192129], [-0.5463474, -0.1897013, -1.6109908, 1.1600197, 0.2355175, 0.7431571],
                      [-0.3496264, -0.0522738, -1.6833475, 1.2487445, 0.1296605, 0.6531504], [-0.2299080, -0.1841171, -1.7537270, 1.3589362, 0.1654926, 0.8529139],
                      [-0.1403695, -0.4164103, -1.8408963, 1.3716500, 0.2060994, 0.9893460], [-0.2332071, -0.5936512, -1.8870399, 1.4552834, 0.0195072, 1.0669760],
                      [-0.2300239, -1.4116396, -1.8819211, 1.1254821, 0.3495971, 1.3864237], [-0.0187657, -1.5084414, -1.8875910, 1.5262295, 0.8092253, 1.9199706],
                      [0.0066601, -1.5512746, -1.6101097, 1.4239284, 0.8035201, 1.9149790], [-0.1446207, -1.9825488, -0.8208647, 1.4516564, 1.1849712, 1.8480887],
                      [-0.6108302, -1.9348916, -1.0109171, 0.8598719, 1.4121891, 1.8276713], [-0.7901841, -1.9069933, -1.0716456, 0.6924882, 1.3664749, 1.6751629],
                      [-1.2537752, -2.1546411, -0.7841795, 0.0328592, 1.7392993, 1.6518787], [-1.1640728, -1.9348487, -0.8201570, 0.2030532, 2.1956809, 1.3064180],
                      [-1.0541359, -2.2773923, -0.1245015, -0.1888635, 2.1030945, 1.2355536]])

path = [Node(pathArray[i, 0], pathArray[i, 1], pathArray[i, 2], pathArray[i, 3], pathArray[i, 4], pathArray[i, 5]) for i in range(pathArray.shape[0])]

armState = UR5eStateArmCoppeliaSimAPI()
armStateVirtual = UR5eVirtualArmCoppeliaSimAPI()
armState.start_sim()
pathX, pathY, pathZ, pathP, pathQ, pathR = extract_path_class_6d(path)
pathX = np.array(pathX)
pathY = np.array(pathY)
pathZ = np.array(pathZ)
pathP = np.array(pathP)
pathQ = np.array(pathQ)
pathR = np.array(pathR)
armState.set_goal_joint_value(pathArray[0].reshape(6,1))
armState.set_aux_joint_value(pathArray[-1].reshape(6,1))
armState.set_start_joint_value(pathArray[-2].reshape(6,1))
for i in range(len(pathX)):
    jointVal = np.array([pathX[i], pathY[i], pathZ[i], pathP[i], pathQ[i], pathR[i]]).reshape(6, 1)
    armStateVirtual.set_joint_value(jointVal)
    time.sleep(0.5)
armState.stop_sim()