import sys
import os

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))
sys.path.append("/home/yuth/coppelia/programming/zmqRemoteApi/clients/python")

import numpy as np
from zmqRemoteApi import RemoteAPIClient


class UR5eArmCoppeliaSimAPI:

    def __init__(self, robotChoice="V") -> None:
        # robot choice
        self.robotChoice = {"V":"UR5e_virtual",
                            "D":"UR5e_dynamic",
                            "UR5V":"UR5_virtual",
                            "UR5D":"UR5_dynamic"}
        self.robot = self.robotChoice[robotChoice]

        # coppeliasim
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        self.client.setStepping(True)

        # arm handle
        self.joint1Handle = self.sim.getObject(f'/{self.robot}/shoulder_pan_joint')
        self.joint2Handle = self.sim.getObject(f'/{self.robot}/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint')
        self.joint3Handle = self.sim.getObject(f'/{self.robot}/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint')
        self.joint4Handle = self.sim.getObject(f'/{self.robot}/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint')
        self.joint5Handle = self.sim.getObject(f'/{self.robot}/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint/wrist_1_link_resp/wrist_2_joint')
        self.joint6Handle = self.sim.getObject(f'/{self.robot}/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint/wrist_1_link_resp/wrist_2_joint/wrist_2_link_resp/wrist_3_joint')
        self.jointHandleList = [self.joint1Handle, self.joint2Handle, self.joint3Handle, self.joint4Handle, self.joint5Handle, self.joint6Handle]

        # Start Aux Goal Handle
        self.joint1GoalHandle = self.sim.getObject('/UR5e_Goal/shoulder_pan_joint')
        self.joint2GoalHandle = self.sim.getObject('/UR5e_Goal/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint')
        self.joint3GoalHandle = self.sim.getObject('/UR5e_Goal/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint')
        self.joint4GoalHandle = self.sim.getObject('/UR5e_Goal/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint')
        self.joint5GoalHandle = self.sim.getObject('/UR5e_Goal/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint/wrist_1_link_resp/wrist_2_joint')
        self.joint6GoalHandle = self.sim.getObject('/UR5e_Goal/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint/wrist_1_link_resp/wrist_2_joint/wrist_2_link_resp/wrist_3_joint')
        self.jointGoalHandleList = [self.joint1GoalHandle, self.joint2GoalHandle, self.joint3GoalHandle, self.joint4GoalHandle, self.joint5GoalHandle, self.joint6GoalHandle]

        self.joint1AuxHandle = self.sim.getObject('/UR5e_Aux/shoulder_pan_joint')
        self.joint2AuxHandle = self.sim.getObject('/UR5e_Aux/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint')
        self.joint3AuxHandle = self.sim.getObject('/UR5e_Aux/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint')
        self.joint4AuxHandle = self.sim.getObject('/UR5e_Aux/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint')
        self.joint5AuxHandle = self.sim.getObject('/UR5e_Aux/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint/wrist_1_link_resp/wrist_2_joint')
        self.joint6AuxHandle = self.sim.getObject('/UR5e_Aux/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint/wrist_1_link_resp/wrist_2_joint/wrist_2_link_resp/wrist_3_joint')
        self.jointAuxHandleList = [self.joint1AuxHandle, self.joint2AuxHandle, self.joint3AuxHandle, self.joint4AuxHandle, self.joint5AuxHandle, self.joint6AuxHandle]

        self.joint1StartHandle = self.sim.getObject('/UR5e_Start/shoulder_pan_joint')
        self.joint2StartHandle = self.sim.getObject('/UR5e_Start/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint')
        self.joint3StartHandle = self.sim.getObject('/UR5e_Start/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint')
        self.joint4StartHandle = self.sim.getObject('/UR5e_Start/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint')
        self.joint5StartHandle = self.sim.getObject('/UR5e_Start/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint/wrist_1_link_resp/wrist_2_joint')
        self.joint6StartHandle = self.sim.getObject('/UR5e_Start/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint/wrist_1_link_resp/wrist_2_joint/wrist_2_link_resp/wrist_3_joint')
        self.jointStartHandleList = [self.joint1StartHandle, self.joint2StartHandle, self.joint3StartHandle, self.joint4StartHandle, self.joint5StartHandle, self.joint6StartHandle]

        # arm collision
        self.robotBase = self.sim.getObject(f'/{self.robot}')
        self.collection = self.sim.createCollection(0)
        self.sim.addItemToCollection(self.collection, self.sim.handle_tree, self.robotBase, 0)

    def start_sim(self):
        self.sim.startSimulation()

    def stop_sim(self):
        self.sim.stopSimulation()

    def step_sim(self):
        self.client.step()

    def collision_check(self, jointValue):
        self.set_joint_value(jointValue)
        result, _ = self.sim.checkCollision(self.collection, self.sim.handle_all)
        if result == 1:
            return True
        else:
            return False

    def set_joint_value(self, jointValue):
        if isinstance(jointValue, np.ndarray):
            for i in range(jointValue.shape[0]):
                self.sim.setJointPosition(self.jointHandleList[i], jointValue[i, 0])
        else:
            for i in range(jointValue.config.shape[0]):
                self.sim.setJointPosition(self.jointHandleList[i], jointValue.config[i, 0])

    def set_goal_joint_value(self, jointValue):
        if isinstance(jointValue, np.ndarray):
            for i in range(jointValue.shape[0]):
                self.sim.setJointPosition(self.jointGoalHandleList[i], jointValue[i, 0])
        else:
            for i in range(jointValue.config.shape[0]):
                self.sim.setJointPosition(self.jointGoalHandleList[i], jointValue.config[i, 0])

    def set_aux_joint_value(self, jointValue):
        if isinstance(jointValue, np.ndarray):
            for i in range(jointValue.shape[0]):
                self.sim.setJointPosition(self.jointAuxHandleList[i], jointValue[i, 0])
        else:
            for i in range(jointValue.config.shape[0]):
                self.sim.setJointPosition(self.jointAuxHandleList[i], jointValue.config[i, 0])

    def set_start_joint_value(self, jointValue):
        if isinstance(jointValue, np.ndarray):
            for i in range(jointValue.shape[0]):
                self.sim.setJointPosition(self.jointStartHandleList[i], jointValue[i, 0])
        else:
            for i in range(jointValue.config.shape[0]):
                self.sim.setJointPosition(self.jointStartHandleList[i], jointValue.config[i, 0])

    def set_joint_target_value(self, jointValue):
        for i in range(jointValue.config.shape[0]):
            self.sim.setJointTargetPosition(self.jointHandleList[i], jointValue.config[i, 0])


if __name__ == "__main__":
    from datasave.joint_value.pre_record_value import wrap_to_pi, newThetaInit, newThetaApp, newThetaGoal

    # UR5e scene
    # armVirtual = UR5eVirtualArmCoppeliaSimAPI()
    # armVirtual.start_sim()
    # angle = np.linspace(-np.pi, np.pi, 360)

    # # loop in simulation
    # for ang in angle:
    #     jointValue = np.array([ang, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(6,1)
    #     armVirtual.set_joint_value(jointValue)
    #     # triggers next simulation step
    #     armVirtual.step_sim()

    # # stop simulation
    # armVirtual.stop_sim()

    # View State
    armState = UR5eArmCoppeliaSimAPI()
    # armState.start_sim()
    jointV = np.array([0.0,0.0,0.0,0.0,0.0,0.0]).reshape(6,1)
    armState.set_joint_value(jointV)
    armState.set_goal_joint_value(jointV)
    armState.set_aux_joint_value(jointV)
    armState.set_start_joint_value(jointV)
