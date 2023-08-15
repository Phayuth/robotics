import sys
sys.path.append("/home/yuth/coppelia/programming/zmqRemoteApi/clients/python")

import numpy as np
from zmqRemoteApi import RemoteAPIClient
import time

# create client and sim object
client = RemoteAPIClient()
sim = client.getObject('sim')

# create handle
joint1Handle = sim.getObject('/UR5e_virtual/shoulder_pan_joint')
joint2Handle = sim.getObject('/UR5e_virtual/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint')
joint3Handle = sim.getObject('/UR5e_virtual/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint')
joint4Handle = sim.getObject('/UR5e_virtual/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint')
joint5Handle = sim.getObject('/UR5e_virtual/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint/wrist_1_link_resp/wrist_2_joint')
joint6Handle = sim.getObject('/UR5e_virtual/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint/wrist_1_link_resp/wrist_2_joint/wrist_2_link_resp/wrist_3_joint')

robotBase = sim.getObject('/UR5e_virtual')
collection = sim.createCollection(0)
sim.addItemToCollection(collection, sim.handle_tree, robotBase, 0)

# Run a simulation in stepping mode:
client.setStepping(True)
sim.startSimulation()

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

    collisionTimePre = time.perf_counter_ns()
    sim.setJointPosition(joint1Handle, x)
    sim.setJointPosition(joint2Handle, y)
    sim.setJointPosition(joint3Handle, z)
    sim.setJointPosition(joint4Handle, p)
    sim.setJointPosition(joint5Handle, q)
    sim.setJointPosition(joint6Handle, r)
    result, pairHandles = sim.checkCollision(collection, sim.handle_all)
    collisionTimeAfter = time.perf_counter_ns()
    print(f"{collisionTimeAfter - collisionTimePre} nanosec")
    if result == 1:
        print(True)
    else:
        print(False)
    # triggers next simulation step
    # client.step()

sim.stopSimulation()