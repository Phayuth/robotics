import sys

sys.path.append("/home/yuth/coppelia/programming/zmqRemoteApi/clients/python")
import numpy as np
from zmqRemoteApi import RemoteAPIClient

print('Program started')

# create client and sim object
client = RemoteAPIClient()
sim = client.getObject('sim')
simIK = client.getObject('simIK')

# simBase=sim.getObject('/IRB140')
# simTip=sim.getObject('/IRB140/joint/link/joint/link/joint/link/joint/link/joint/link/joint/link/tip')
# simTarget=sim.getObject('/IRB140/link1_visible/manipulationSphereBase/manipulationSphere/target')

simBase=sim.getObject('/UR5e_dynamic')
simTip=sim.getObject('/UR5e_dynamic/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_respondable/elbow_joint/forearm_link_resp/wrist_1_joint/wrist_1_link_resp/wrist_2_joint/wrist_2_link_resp/wrist_3_joint/wrist_3_link_resp/tip')
simTarget=sim.getObject('/UR5e_dynamic/target')

ikEnv=simIK.createEnvironment()

ikGroup_undamped=simIK.createIkGroup(ikEnv)
simIK.setIkGroupCalculation(ikEnv,ikGroup_undamped,simIK.method_pseudo_inverse,0,10)
simIK.addIkElementFromScene(ikEnv,ikGroup_undamped,simBase,simTip,simTarget,simIK.constraint_pose)
ikGroup_damped=simIK.createIkGroup(ikEnv)
simIK.setIkGroupCalculation(ikEnv,ikGroup_damped,simIK.method_damped_least_squares,0.3,99)
simIK.addIkElementFromScene(ikEnv,ikGroup_damped,simBase,simTip,simTarget,simIK.constraint_pose)

client.setStepping(True)
sim.startSimulation()
while True:
    if simIK.applyIkEnvironmentToScene(ikEnv,ikGroup_undamped,True) != simIK.result_success:
        simIK.applyIkEnvironmentToScene(ikEnv,ikGroup_damped)
        client.step()