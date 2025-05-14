import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

from datasave.joint_value.pre_record_value import PreRecordedPath
from datasave.joint_value.experiment_paper import URHarvesting
import numpy as np
from simulator.sim_ur5e_api import UR5eArmCoppeliaSimAPI

ur = UR5eArmCoppeliaSimAPI()

# IK
if True:
    ur.start_sim()
    try:
        ur.update_joint_position()

        while True:
            # find config
            targets = ur.sim.getObject("/t0")
            targetPose = ur.sim.getObjectPose(
                targets, ur.simBase
            )  # pose: the position and quaternion of the object (x,y,z,qx,qy,qz,qw).

            # if state is None:
            #     state = ur.ik_find_config(targetPose)
            #     ur.set_joint_position(ur.jointDynamicHandles, np.array(state).reshape(6, 1))

            # if state is not None:
            state = ur.ik_solve(targetPose)
            if state is not None:
                ur.set_joint_position(
                    ur.jointDynamicHandles, np.array(state).reshape(6, 1)
                )

            print(f"> state: {state}")

    except KeyboardInterrupt:
        pass

    finally:
        ur.stop_sim()

# Play back
if False:
    ur.start_sim()
    try:
        while True:
            jj = np.array(
                [
                    -2.4307362897230953,
                    -1.9950339217373045,
                    -0.0010261435935210133,
                    -0.6685270374364699,
                    0.9877321445181428,
                    1.2663907399297298,
                ]
            ).reshape(6, 1)
            jg = np.array(
                [
                    -0.0027387777911584976,
                    -1.9624139271178187,
                    -1.4210033416748047,
                    -2.6216727695860804,
                    -1.4972699324237269,
                    -3.134235207234518,
                ]
            ).reshape(6, 1)
            ur.set_joint_position(ur.jointDynamicHandles, jg)
            # # set joint visualize
            # q = URHarvesting.PoseSingle1()
            # qS = q.xStart
            # qA = q.xApp
            # qG = q.xGoal
            # ur.set_pose_aux_goal(qA, qG)

            # # play back
            # path = PreRecordedPath.path.T  # shape(6, 12)
            # ur.play_back_path(path)

    except KeyboardInterrupt:
        pass

    finally:
        ur.stop_sim()
