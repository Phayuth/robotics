import sys

sys.path.append("/home/yuth/coppelia/programming/zmqRemoteApi/clients/python")
import numpy as np
from zmqRemoteApi import RemoteAPIClient

print('Program started')

# create client and sim object
client = RemoteAPIClient()
sim = client.getObject('sim')

# create handle
joint1Handle = sim.getObject('/UR5_virtual/joint')
joint2Handle = sim.getObject('/UR5_virtual/link/joint')
joint3Handle = sim.getObject('/UR5_virtual/link/joint/link/joint')
joint4Handle = sim.getObject('/UR5_virtual/link/joint/link/joint/link/joint')
joint5Handle = sim.getObject('/UR5_virtual/link/joint/link/joint/link/joint/link/joint')
joint6Handle = sim.getObject('/UR5_virtual/link/joint/link/joint/link/joint/link/joint/link/joint')

robotBase = sim.getObject('/UR5_virtual')
collection = sim.createCollection(0)
sim.addItemToCollection(collection, sim.handle_tree, robotBase, 0)

# Run a simulation in stepping mode:
client.setStepping(True)
sim.startSimulation()

# set joint cmd
# target joint configuration
# sim.setJointTargetPosition(joint1Handle, -1.57)
# sim.setJointTargetPosition(joint2Handle, -2.3)
# sim.setJointTargetPosition(joint3Handle, -1.4)
# sim.setJointTargetPosition(joint4Handle, 0.1)
# sim.setJointTargetPosition(joint5Handle, 1.57)
# sim.setJointTargetPosition(joint6Handle, 0)

# sim.setJointPosition(joint1Handle, -1.57)
# sim.setJointPosition(joint2Handle, -2.3)
# sim.setJointPosition(joint3Handle, -1.4)
# sim.setJointPosition(joint4Handle, 0.1)
# sim.setJointPosition(joint5Handle, 1.57)
# sim.setJointPosition(joint6Handle, 0)

angle = np.linspace(-np.pi, np.pi, 360)

# loop in simulation
# while (t := sim.getSimulationTime()) < 10:
for i in range(len(angle)):
    # do something
    x = np.random.uniform(low=-np.pi, high=np.pi)
    y = np.random.uniform(low=-np.pi, high=np.pi)
    z = np.random.uniform(low=-np.pi, high=np.pi)
    p = np.random.uniform(low=-np.pi, high=np.pi)
    q = np.random.uniform(low=-np.pi, high=np.pi)
    r = np.random.uniform(low=-np.pi, high=np.pi)

    sim.setJointPosition(joint1Handle, angle[i])

    # sim.setJointPosition(joint1Handle, x)
    # sim.setJointPosition(joint2Handle, y)
    # sim.setJointPosition(joint3Handle, z)
    # sim.setJointPosition(joint4Handle, p)
    # sim.setJointPosition(joint5Handle, q)
    # sim.setJointPosition(joint6Handle, r)

    result, pairHandles = sim.checkCollision(collection, sim.handle_all)
    # print(f"collision status is {result} at simulationTime is  sec")
    if result == 1:
        print(True)
    else:
        print(False)
    # triggers next simulation step
    # client.step()

# stop simulation
sim.stopSimulation()
print('Program ended')