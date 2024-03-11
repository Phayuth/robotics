import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from simulator.sim_ur5e_api import UR5eArmCoppeliaSimAPI

ur = UR5eArmCoppeliaSimAPI()

ur.start_sim()
try:
    ur.update_joint_position()

    while True:
        # find config
        targets = ur.sim.getObject('/t0')
        targetPose = ur.sim.getObjectPose(targets, ur.simBase)  # pose: the position and quaternion of the object (x,y,z,qx,qy,qz,qw).

        # if state is None:
        #     state = ur.ik_find_config(targetPose)
        #     ur.set_joint_position(ur.jointDynamicHandles, np.array(state).reshape(6, 1))

        # if state is not None:
        state = ur.ik_solve(targetPose)
        if state is not None:
            ur.set_joint_position(ur.jointDynamicHandles, np.array(state).reshape(6, 1))

        print(f"> state: {state}")

except KeyboardInterrupt:
    pass

finally:
    ur.stop_sim()