import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
from simulator.sim_ur5e_api import UR5eArmCoppeliaSimAPI

ur = UR5eArmCoppeliaSimAPI()


def move_config():
    movement = np.deg2rad(
        [
            [0, -90, -90, 0, 0, 0],
            [90, -90, -90, 90, 90, 90],
            [-90, -45, 90, 135, 90, 90],
            [0, -90, -90, 0, 0, 0],
        ]
    ).tolist()
    movementVelo = [None, None, None, None]
    ur.move_to_config_sequences(movement, movementVelo)


def move_pose():  # pick and place
    pgh = ur.sim.getObjectHandle("/pnpt1")
    gh = ur.sim.getObjectHandle("/pnpt2")
    dh = ur.sim.getObjectHandle("/pnpt3")

    pgp = ur.sim.getObjectPose(pgh, ur.simBase)  # pose: (x,y,z,qx,qy,qz,qw).
    gp = ur.sim.getObjectPose(gh, ur.simBase)
    dp = ur.sim.getObjectPose(dh, ur.simBase)

    movement = [pgp, gp, pgp, dp]
    qSequence, qKnot = ur.move_to_pose_sequences(movement)


def scan_pose():
    movement = np.deg2rad([[-180, -90, -90, -90, 90, 0]]).tolist()
    movementVelo = [None]
    ur.move_to_config_sequences(movement, movementVelo)
    ur.update_joint_position()

    scanObjH = [ur.sim.getObjectHandle(f"/s{i}") for i in range(10)]
    scanPose = [ur.sim.getObjectPose(soh, ur.simBase) for soh in scanObjH]
    qSequence, qKnot = ur.move_to_pose_sequences(scanPose)
    ur.update_joint_position()


ur.start_sim()

try:
    # movement = np.deg2rad([[-90, -45, -90, -45, 0, 0]]).tolist()
    # movementVelo = [None]
    # ur.move_to_config_sequences(movement, movementVelo)
    ur.update_joint_position()
    while True:
        # move_config() # config sequences
        # move_pose() # pose sequences
        scan_pose()

except KeyboardInterrupt:
    pass

finally:
    ur.stop_sim()
