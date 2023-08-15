import sys
import os

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))
sys.path.append("/home/yuth/coppelia/programming/zmqRemoteApi/clients/python")

import numpy as np
from zmqRemoteApi import RemoteAPIClient


class UR5VirtualArmCoppeliaSimAPI:

    def __init__(self) -> None:
        # coppeliasim
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        self.client.setStepping(True)

        # arm handle
        self.joint1Handle = self.sim.getObject('/UR5_virtual/joint')
        self.joint2Handle = self.sim.getObject('/UR5_virtual/link/joint')
        self.joint3Handle = self.sim.getObject('/UR5_virtual/link/joint/link/joint')
        self.joint4Handle = self.sim.getObject('/UR5_virtual/link/joint/link/joint/link/joint')
        self.joint5Handle = self.sim.getObject('/UR5_virtual/link/joint/link/joint/link/joint/link/joint')
        self.joint6Handle = self.sim.getObject('/UR5_virtual/link/joint/link/joint/link/joint/link/joint/link/joint')

        # arm collision
        self.robotBase = self.sim.getObject('/UR5_virtual')
        self.collection = self.sim.createCollection(0)
        self.sim.addItemToCollection(self.collection, self.sim.handle_tree, self.robotBase, 0)

    def start_sim(self):
        self.sim.startSimulation()

    def stop_sim(self):
        self.sim.stopSimulation()

    def step_sim(self):
        self.client.step()

    def collsion_check(self, jointValue):
        self.set_joint_value(jointValue)
        result, pairHandles = self.sim.checkCollision(self.collection, self.sim.handle_all)
        if result == 1:
            return True
        else:
            return False

    def set_joint_value(self, jointValue):
        if isinstance(jointValue, np.ndarray):
            x, y, z, p, q, r = jointValue[0, 0], jointValue[1, 0], jointValue[2, 0], jointValue[3, 0], jointValue[4, 0], jointValue[5, 0]
        else:
            x, y, z, p, q, r = jointValue.x, jointValue.y, jointValue.z, jointValue.p, jointValue.q, jointValue.r
        self.sim.setJointPosition(self.joint1Handle, x)
        self.sim.setJointPosition(self.joint2Handle, y)
        self.sim.setJointPosition(self.joint3Handle, z)
        self.sim.setJointPosition(self.joint4Handle, p)
        self.sim.setJointPosition(self.joint5Handle, q)
        self.sim.setJointPosition(self.joint6Handle, r)


class UR5DynamicArmCoppeliaSimAPI:

    def __init__(self) -> None:
        # coppeliasim
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        self.client.setStepping(True)

        # arm handle
        self.joint1Handle = self.sim.getObject('/UR5_dynamic/joint')
        self.joint2Handle = self.sim.getObject('/UR5_dynamic/link/joint')
        self.joint3Handle = self.sim.getObject('/UR5_dynamic/link/joint/link/joint')
        self.joint4Handle = self.sim.getObject('/UR5_dynamic/link/joint/link/joint/link/joint')
        self.joint5Handle = self.sim.getObject('/UR5_dynamic/link/joint/link/joint/link/joint/link/joint')
        self.joint6Handle = self.sim.getObject('/UR5_dynamic/link/joint/link/joint/link/joint/link/joint/link/joint')

        # arm collision
        self.robotBase = self.sim.getObject('/UR5_dynamic')
        self.collection = self.sim.createCollection(0)
        self.sim.addItemToCollection(self.collection, self.sim.handle_tree, self.robotBase, 0)

    def start_sim(self):
        self.sim.startSimulation()

    def stop_sim(self):
        self.sim.stopSimulation()

    def step_sim(self):
        self.client.step()

    def collsion_check(self, jointValue):
        self.set_joint_value(jointValue)
        result, pairHandles = self.sim.checkCollision(self.collection, self.sim.handle_all)
        if result == 1:
            return True
        else:
            return False

    def set_joint_value(self, jointValue):
        if isinstance(jointValue, np.ndarray):
            x, y, z, p, q, r = jointValue[0, 0], jointValue[1, 0], jointValue[2, 0], jointValue[3, 0], jointValue[4, 0], jointValue[5, 0]
        else:
            x, y, z, p, q, r = jointValue.x, jointValue.y, jointValue.z, jointValue.p, jointValue.q, jointValue.r
        self.sim.setJointPosition(self.joint1Handle, x)
        self.sim.setJointPosition(self.joint2Handle, y)
        self.sim.setJointPosition(self.joint3Handle, z)
        self.sim.setJointPosition(self.joint4Handle, p)
        self.sim.setJointPosition(self.joint5Handle, q)
        self.sim.setJointPosition(self.joint6Handle, r)

    def set_joint_target_value(self, jointValue):
        if isinstance(jointValue, np.ndarray):
            x, y, z, p, q, r = jointValue[0, 0], jointValue[1, 0], jointValue[2, 0], jointValue[3, 0], jointValue[4, 0], jointValue[5, 0]
        else:
            x, y, z, p, q, r = jointValue.x, jointValue.y, jointValue.z, jointValue.p, jointValue.q, jointValue.r
        self.sim.setJointTargetPosition(self.joint1Handle, x)
        self.sim.setJointTargetPosition(self.joint2Handle, y)
        self.sim.setJointTargetPosition(self.joint3Handle, z)
        self.sim.setJointTargetPosition(self.joint4Handle, p)
        self.sim.setJointTargetPosition(self.joint5Handle, q)
        self.sim.setJointTargetPosition(self.joint6Handle, r)


class UR5eVirtualArmCoppeliaSimAPI:

    def __init__(self) -> None:
        # coppeliasim
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        self.client.setStepping(True)

        # arm handle
        self.joint1Handle = self.sim.getObject('/UR5e_virtual/shoulder_pan_joint')
        self.joint2Handle = self.sim.getObject('/UR5e_virtual/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint')
        self.joint3Handle = self.sim.getObject('/UR5e_virtual/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint')
        self.joint4Handle = self.sim.getObject('/UR5e_virtual/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint')
        self.joint5Handle = self.sim.getObject('/UR5e_virtual/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint/wrist_1_link_resp/wrist_2_joint')
        self.joint6Handle = self.sim.getObject('/UR5e_virtual/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint/wrist_1_link_resp/wrist_2_joint/wrist_2_link_resp/wrist_3_joint')

        # arm collision
        self.robotBase = self.sim.getObject('/UR5e_virtual')
        self.collection = self.sim.createCollection(0)
        self.sim.addItemToCollection(self.collection, self.sim.handle_tree, self.robotBase, 0)

    def start_sim(self):
        self.sim.startSimulation()

    def stop_sim(self):
        self.sim.stopSimulation()

    def step_sim(self):
        self.client.step()

    def collsion_check(self, jointValue):
        self.set_joint_value(jointValue)
        result, pairHandles = self.sim.checkCollision(self.collection, self.sim.handle_all)
        if result == 1:
            return True
        else:
            return False

    def set_joint_value(self, jointValue):
        if isinstance(jointValue, np.ndarray):
            x, y, z, p, q, r = jointValue[0, 0], jointValue[1, 0], jointValue[2, 0], jointValue[3, 0], jointValue[4, 0], jointValue[5, 0]
        else:
            x, y, z, p, q, r = jointValue.x, jointValue.y, jointValue.z, jointValue.p, jointValue.q, jointValue.r
        self.sim.setJointPosition(self.joint1Handle, x)
        self.sim.setJointPosition(self.joint2Handle, y)
        self.sim.setJointPosition(self.joint3Handle, z)
        self.sim.setJointPosition(self.joint4Handle, p)
        self.sim.setJointPosition(self.joint5Handle, q)
        self.sim.setJointPosition(self.joint6Handle, r)


class UR5eDynamicArmCoppeliaSimAPI:

    def __init__(self) -> None:
        # coppeliasim
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        self.client.setStepping(True)

        # arm handle
        self.joint1Handle = self.sim.getObject('/UR5e_dynamic/shoulder_pan_joint')
        self.joint2Handle = self.sim.getObject('/UR5e_dynamic/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint')
        self.joint3Handle = self.sim.getObject('/UR5e_dynamic/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint')
        self.joint4Handle = self.sim.getObject('/UR5e_dynamic/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint')
        self.joint5Handle = self.sim.getObject('/UR5e_dynamic/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint/wrist_1_link_resp/wrist_2_joint')
        self.joint6Handle = self.sim.getObject('/UR5e_dynamic/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint/wrist_1_link_resp/wrist_2_joint/wrist_2_link_resp/wrist_3_joint')

        # arm collision
        self.robotBase = self.sim.getObject('/UR5e_dynamic')
        self.collection = self.sim.createCollection(0)
        self.sim.addItemToCollection(self.collection, self.sim.handle_tree, self.robotBase, 0)

    def start_sim(self):
        self.sim.startSimulation()

    def stop_sim(self):
        self.sim.stopSimulation()

    def step_sim(self):
        self.client.step()

    def collsion_check(self, jointValue):
        self.set_joint_value(jointValue)
        result, pairHandles = self.sim.checkCollision(self.collection, self.sim.handle_all)
        if result == 1:
            return True
        else:
            return False

    def set_joint_value(self, jointValue):
        if isinstance(jointValue, np.ndarray):
            x, y, z, p, q, r = jointValue[0, 0], jointValue[1, 0], jointValue[2, 0], jointValue[3, 0], jointValue[4, 0], jointValue[5, 0]
        else:
            x, y, z, p, q, r = jointValue.x, jointValue.y, jointValue.z, jointValue.p, jointValue.q, jointValue.r
        self.sim.setJointPosition(self.joint1Handle, x)
        self.sim.setJointPosition(self.joint2Handle, y)
        self.sim.setJointPosition(self.joint3Handle, z)
        self.sim.setJointPosition(self.joint4Handle, p)
        self.sim.setJointPosition(self.joint5Handle, q)
        self.sim.setJointPosition(self.joint6Handle, r)

    def set_joint_target_value(self, jointValue):
        if isinstance(jointValue, np.ndarray):
            x, y, z, p, q, r = jointValue[0, 0], jointValue[1, 0], jointValue[2, 0], jointValue[3, 0], jointValue[4, 0], jointValue[5, 0]
        else:
            x, y, z, p, q, r = jointValue.x, jointValue.y, jointValue.z, jointValue.p, jointValue.q, jointValue.r
        self.sim.setJointTargetPosition(self.joint1Handle, x)
        self.sim.setJointTargetPosition(self.joint2Handle, y)
        self.sim.setJointTargetPosition(self.joint3Handle, z)
        self.sim.setJointTargetPosition(self.joint4Handle, p)
        self.sim.setJointTargetPosition(self.joint5Handle, q)
        self.sim.setJointTargetPosition(self.joint6Handle, r)


class UR5eStateArmCoppeliaSimAPI:

    def __init__(self) -> None:
        # coppeliasim
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        self.client.setStepping(True)

        # arm handle Goal
        self.joint1GoalHandle = self.sim.getObject('/UR5e_Goal/shoulder_pan_joint')
        self.joint2GoalHandle = self.sim.getObject('/UR5e_Goal/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint')
        self.joint3GoalHandle = self.sim.getObject('/UR5e_Goal/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint')
        self.joint4GoalHandle = self.sim.getObject('/UR5e_Goal/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint')
        self.joint5GoalHandle = self.sim.getObject('/UR5e_Goal/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint/wrist_1_link_resp/wrist_2_joint')
        self.joint6GoalHandle = self.sim.getObject('/UR5e_Goal/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint/wrist_1_link_resp/wrist_2_joint/wrist_2_link_resp/wrist_3_joint')

        # arm handle Aux
        self.joint1AuxHandle = self.sim.getObject('/UR5e_Aux/shoulder_pan_joint')
        self.joint2AuxHandle = self.sim.getObject('/UR5e_Aux/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint')
        self.joint3AuxHandle = self.sim.getObject('/UR5e_Aux/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint')
        self.joint4AuxHandle = self.sim.getObject('/UR5e_Aux/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint')
        self.joint5AuxHandle = self.sim.getObject('/UR5e_Aux/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint/wrist_1_link_resp/wrist_2_joint')
        self.joint6AuxHandle = self.sim.getObject('/UR5e_Aux/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint/wrist_1_link_resp/wrist_2_joint/wrist_2_link_resp/wrist_3_joint')

        # arm handle Start
        self.joint1StartHandle = self.sim.getObject('/UR5e_Start/shoulder_pan_joint')
        self.joint2StartHandle = self.sim.getObject('/UR5e_Start/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint')
        self.joint3StartHandle = self.sim.getObject('/UR5e_Start/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint')
        self.joint4StartHandle = self.sim.getObject('/UR5e_Start/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint')
        self.joint5StartHandle = self.sim.getObject('/UR5e_Start/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint/wrist_1_link_resp/wrist_2_joint')
        self.joint6StartHandle = self.sim.getObject('/UR5e_Start/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint/wrist_1_link_resp/wrist_2_joint/wrist_2_link_resp/wrist_3_joint')

    def start_sim(self):
        self.sim.startSimulation()

    def stop_sim(self):
        self.sim.stopSimulation()

    def step_sim(self):
        self.client.step()

    def set_goal_joint_value(self, jointValue):
        if isinstance(jointValue, np.ndarray):
            x, y, z, p, q, r = jointValue[0, 0], jointValue[1, 0], jointValue[2, 0], jointValue[3, 0], jointValue[4, 0], jointValue[5, 0]
        else:
            x, y, z, p, q, r = jointValue.x, jointValue.y, jointValue.z, jointValue.p, jointValue.q, jointValue.r
        self.sim.setJointPosition(self.joint1GoalHandle, x)
        self.sim.setJointPosition(self.joint2GoalHandle, y)
        self.sim.setJointPosition(self.joint3GoalHandle, z)
        self.sim.setJointPosition(self.joint4GoalHandle, p)
        self.sim.setJointPosition(self.joint5GoalHandle, q)
        self.sim.setJointPosition(self.joint6GoalHandle, r)

    def set_aux_joint_value(self, jointValue):
        if isinstance(jointValue, np.ndarray):
            x, y, z, p, q, r = jointValue[0, 0], jointValue[1, 0], jointValue[2, 0], jointValue[3, 0], jointValue[4, 0], jointValue[5, 0]
        else:
            x, y, z, p, q, r = jointValue.x, jointValue.y, jointValue.z, jointValue.p, jointValue.q, jointValue.r
        self.sim.setJointPosition(self.joint1AuxHandle, x)
        self.sim.setJointPosition(self.joint2AuxHandle, y)
        self.sim.setJointPosition(self.joint3AuxHandle, z)
        self.sim.setJointPosition(self.joint4AuxHandle, p)
        self.sim.setJointPosition(self.joint5AuxHandle, q)
        self.sim.setJointPosition(self.joint6AuxHandle, r)

    def set_start_joint_value(self, jointValue):
        if isinstance(jointValue, np.ndarray):
            x, y, z, p, q, r = jointValue[0, 0], jointValue[1, 0], jointValue[2, 0], jointValue[3, 0], jointValue[4, 0], jointValue[5, 0]
        else:
            x, y, z, p, q, r = jointValue.x, jointValue.y, jointValue.z, jointValue.p, jointValue.q, jointValue.r
        self.sim.setJointPosition(self.joint1StartHandle, x)
        self.sim.setJointPosition(self.joint2StartHandle, y)
        self.sim.setJointPosition(self.joint3StartHandle, z)
        self.sim.setJointPosition(self.joint4StartHandle, p)
        self.sim.setJointPosition(self.joint5StartHandle, q)
        self.sim.setJointPosition(self.joint6StartHandle, r)

if __name__ == "__main__":
    from target_localization.pre_record_value import thetaInit, thetaGoal5, thetaApp5, qCurrent, qAux, qGoal, wrap_to_pi

    # # UR5e scene
    # thetaInit = thetaInit
    # thetaGoal = thetaGoal5
    # thetaApp =  thetaApp5

    # armVirtual = UR5eVirtualArmCoppeliaSimAPI()
    # armVirtual.start_sim()

    # # loop in simulation
    # while (t := armVirtual.sim.getSimulationTime()) < 10:

    #     # do something
    #     if 0 <= t < 4:
    #         armVirtual.set_joint_value(thetaInit)

    #     elif 4 <= t < 7:
    #         armVirtual.set_joint_value(thetaApp)

    #     else:
    #         armVirtual.set_joint_value(thetaGoal)

    #     # triggers next simulation step
    #     armVirtual.step_sim()

    # # stop simulation
    # armVirtual.stop_sim()

    armState = UR5eStateArmCoppeliaSimAPI()
    armState.start_sim()
    armState.set_goal_joint_value(wrap_to_pi(qGoal))
    armState.set_aux_joint_value(wrap_to_pi(qAux))
    armState.set_start_joint_value(wrap_to_pi(thetaInit))
