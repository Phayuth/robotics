import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
import pybullet as p
import pybullet_data

p.setAdditionalSearchPath(pybullet_data.getDataPath())


class UR5eBullet:

    def __init__(self, mode="gui") -> None:
        # connect
        if mode == "gui":
            p.connect(p.GUI)
            # p.connect(p.SHARED_MEMORY_GUI)
        if mode == "no_gui":
            p.connect(p.DIRECT)  # for non-graphical
            # p.connect(p.SHARED_MEMORY)

        # load model and properties
        self.load_model()
        self.numJoints = self.get_num_joints()
        self.jointNames = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.jointIDs = [1, 2, 3, 4, 5, 6]
        self.gripperlinkid = 9

        # inverse kinematic
        self.lower_limits = [-np.pi] * 6
        self.upper_limits = [np.pi] * 6
        self.joint_ranges = [2 * np.pi] * 6
        self.rest_poses = [0, -np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, 0]
        self.joint_damp = [0.01] * 6

    def load_model(self):
        self.planeID = p.loadURDF("plane.urdf", [0, 0, 0])
        self.ur5eID = p.loadURDF("./datasave/urdf/ur5e_extract_calibrated.urdf", [0, 0, 0], useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
        # self.gripper = p.loadURDF("./datasave/urdf/robotiq85.urdf", [0, 0, 0])
        self.tableID = p.loadURDF("table/table.urdf", [0, 0, 0])

    def get_num_joints(self):
        return p.getNumJoints(self.ur5eID)

    def get_joint_link_info(self):
        for i in range(self.numJoints):
            (
                jointIndex,
                jointName,
                jointType,
                qIndex,
                uIndex,
                flags,
                jointDamping,
                jointFriction,
                jointLowerLimit,
                jointUpperLimit,
                jointMaxForce,
                jointMaxVelocity,
                linkName,
                jointAxis,
                parentFramePos,
                parentFrameOrn,
                parentIndex,
            ) = p.getJointInfo(self.ur5eID, i)

            print(f"> ---------------------------------------------<")
            print(f"> jointIndex: {jointIndex}")
            print(f"> jointName: {jointName}")
            print(f"> jointType: {jointType}")
            print(f"> qIndex: {qIndex}")
            print(f"> uIndex: {uIndex}")
            print(f"> flags: {flags}")
            print(f"> jointDamping: {jointDamping}")
            print(f"> jointFriction: {jointFriction}")
            print(f"> jointLowerLimit: {jointLowerLimit}")
            print(f"> jointUpperLimit: {jointUpperLimit}")
            print(f"> jointMaxForce: {jointMaxForce}")
            print(f"> jointMaxVelocity: {jointMaxVelocity}")
            print(f"> linkName: {linkName}")
            print(f"> jointAxis: {jointAxis}")
            print(f"> parentFramePos: {parentFramePos}")
            print(f"> parentFrameOrn: {parentFrameOrn}")
            print(f"> parentIndex: {parentIndex}")

    def control_single_motor(self, jointIndex, jointPosition, jointVelocity=0):
        p.setJointMotorControl2(
            bodyIndex=self.ur5eID,
            jointIndex=jointIndex,
            controlMode=p.POSITION_CONTROL,
            targetPosition=jointPosition,
            targetVelocity=jointVelocity,
            positionGain=0.03,
        )

    def control_array_motors(self, jointPositions, jointVelocities=[0, 0, 0, 0, 0, 0]):
        p.setJointMotorControlArray(
            bodyIndex=self.ur5eID,
            jointIndices=self.jointIDs,
            controlMode=p.POSITION_CONTROL,
            targetPositions=jointPositions,
            targetVelocities=jointVelocities,
            positionGains=[0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
        )

    def get_single_joint_state(self):
        jointPosition, jointVelocity, jointReactionForce, appliedJointMotorTorque = p.getJointState(self.ur5eID, jointIndex=1)
        return jointPosition, jointVelocity, jointReactionForce, appliedJointMotorTorque

    def get_array_joint_state(self):
        j1, j2, j3, j4, j5, j6 = p.getJointStates(self.ur5eID, jointIndices=self.jointIDs)
        return j1, j2, j3, j4, j5, j6

    def get_array_joint_positions(self):
        j1, j2, j3, j4, j5, j6 = self.get_array_joint_state()
        return (j1[0], j2[0], j3[0], j4[0], j5[0], j6[0])

    def forward_kin(self):
        position, orientation = p.getLinkState(self.ur5eID, self.gripperlinkid, computeForwardKinematics=True)
        return position, orientation

    def inverse_kin(self, positions, quaternions):
        joint_angles = p.calculateInverseKinematics(
            self.ur5eID,
            self.gripperlinkid,
            positions,
            quaternions,
            lowerLimits=self.lower_limits,
            upperLimits=self.upper_limits,
            jointRanges=self.joint_ranges,
            jointDamping=self.joint_damp,
            restPoses=self.rest_poses,
        )
        return joint_angles

    def contact_point(self):
        contact_points = p.getContactPoints(bodyA=self.ur5eID, bodyB=self.tableID)
        print(f"> contact_points: {contact_points}")
        # for point in contact_points:
        #     print(f"Contact point details: {point}")

    def closest_point(self):
        closest_points = p.getClosestPoints(bodyA=self.ur5eID, bodyB=self.tableID, distance=0.5)
        print(f"> closest_points: {closest_points}")

    def reset_array_joint_state(self, targetValues):
        for i in range(6):
            p.resetJointState(self.ur5eID, jointIndex=self.jointIDs[i], targetValue=targetValues[i])

    def collisioncheck(self):
        p.performCollisionDetection()

    def move_to_config_sequences(self, seg):
        pass


if __name__ == "__main__":
    import time

    robot = UR5eBullet("gui")
    # robot.get_joint_link_info()
    path = np.loadtxt("/home/yuth/ws_yuthdev/bullet3/build_cmake/examples/MySixJointPlanning/zzz_path.csv", delimiter=",")
    print(f"> path.shape: {path.shape}")

    try:
        for i in range(path.shape[0]):
            # j = robot.get_array_joint_positions()
            # print(f"> j: {j}")
            robot.reset_array_joint_state(path[i])
            # robot.collisioncheck()
            # robot.contact_point()
            # robot.closest_point()
            # robot.control_array_motors(path[i])
            time.sleep(1)
            # p.stepSimulation()

    except KeyboardInterrupt:
        p.disconnect()
