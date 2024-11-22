import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
import pybullet as p
import pybullet_data

p.setAdditionalSearchPath(pybullet_data.getDataPath())
np.set_printoptions(suppress=True, linewidth=1000)


class KUKABullet:

    def __init__(self) -> None:
        p.connect(p.GUI)
        # p.setGravity(0, 0, -9.81)

        self.plane = p.loadURDF("plane.urdf")
        self.kukaId = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
        self.numJoints = p.getNumJoints(self.kukaId)
        self.jointIndices = range(p.getNumJoints(self.kukaId))
        self.kukaEEIndex = self.numJoints - 1

    def getJointStates(self):
        joint_states = p.getJointStates(self.kukaId, self.jointIndices)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques

    def getMotorJointStates(self):
        joint_states = p.getJointStates(self.kukaId, self.jointIndices)
        joint_infos = [p.getJointInfo(self.kukaId, i) for i in self.jointIndices]

        joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques

    def setJointPosition(self, position, kp=1.0, kv=0.3):
        zero_vec = [0.0] * self.numJoints
        p.setJointMotorControlArray(
            self.kukaId,
            self.jointIndices,
            p.POSITION_CONTROL,
            targetPositions=position,
            targetVelocities=zero_vec,
            positionGains=[kp] * self.numJoints,
            velocityGains=[kv] * self.numJoints,
        )

    def compute_mass_matrix(self, q):
        M = p.calculateMassMatrix(self.kukaId, q)
        M = np.array(M)
        return M

    def compute_jacobian(self, q):
        # Get the Jacobians for the CoM of the end-effector link.
        # Note that in this example com_rot = identity, and we would need to use com_rot.T * com_trn.
        # The localPosition is always defined in terms of the link frame coordinates.
        # The relationship between URDF link frame and the center of mass frame (both in world space)
        # urdfLinkFrame = comLinkFrame * localInertialFrame.inverse().
        (
            link_trn,
            link_rot,
            com_trn,
            com_rot,
            frame_pos,
            frame_rot,
            link_vt,
            link_vr,
        ) = p.getLinkState(self.kukaId, self.kukaEEIndex, computeLinkVelocity=1, computeForwardKinematics=1)

        zero_vec = [0.0] * self.numJoints
        jac_t, jac_r = p.calculateJacobian(self.kukaId, self.kukaEEIndex, com_trn, q, zero_vec, zero_vec)
        return jac_t, jac_r

    def _multiplyJacobian(self, jacobian, vector):
        result = [0.0, 0.0, 0.0]
        i = 0
        for c in range(len(vector)):
            if p.getJointInfo(self.kukaId, c)[3] > -1:
                for r in range(3):
                    result[r] += jacobian[r][i] * vector[c]
                i += 1
        return result

    def compute_linear_angular_velocity_ee(self, q_dot):
        # Link linear velocity of CoM  = linearJacobian * q_dot
        # Link angular velocity of CoM = angularJacobian * q_dot
        jac_t, jac_r = self.compute_jacobian()
        link_vt = self._multiplyJacobian(self.kukaId, jac_t, q_dot)
        link_vr = self._multiplyJacobian(self.kukaId, jac_r, q_dot)
        return link_vt, link_vr


if __name__ == "__main__":

    robot = KUKABullet()
    robot.compute_mass_matrix(q=[0.0] * 7)

    try:
        while True:
            p.stepSimulation()

    except KeyboardInterrupt:
        p.disconnect()
