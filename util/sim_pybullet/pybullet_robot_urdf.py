import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import pybullet as p
import time
import math
from datetime import datetime
import pybullet_data

clid = p.connect(p.SHARED_MEMORY)
if (clid < 0):
    p.connect(p.GUI)
    #p.connect(p.SHARED_MEMORY_GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.loadURDF("plane.urdf", [0, 0, -0.3])
ur5eID = p.loadURDF("./datasave/urdf/ur5e_extract_calibrated.urdf", [0, 0, 0])
# p.resetBasePositionAndOrientation(ur5eID, [0, 0, 0], [0, 0, 0, 1])
# kukaEndEffectorIndex = 6
numJoints = p.getNumJoints(ur5eID)
print(f"> numJoints: {numJoints}")

# a = dir(p)
# for e in a:
#     print(e)
#     print("\n")

b = p.getJointStates(ur5eID, [0,1,2,3,4,5,6,7,8,9,10])
print(f"> b: {b}")

i = 0
while 1:
    i += 1

p.disconnect()
