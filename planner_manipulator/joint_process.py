import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from itertools import product


class JointProcess:

    def wrap_to_pi(pose):
        return (pose + np.pi) % (2 * np.pi) - np.pi

    def find_shifted_value(q):  # input must be of shape (n,1)
        shiftedComb = np.array(list(product([-2 * np.pi, 0, 2 * np.pi], repeat=q.shape[0]))).T
        shiftedJointValue = shiftedComb + q
        isInLimitCheck = np.logical_and(shiftedJointValue >= -2 * np.pi, shiftedJointValue <= 2 * np.pi)
        isInLimitMask = np.all(isInLimitCheck, axis=0)
        inLimitJointValue = shiftedJointValue[:, isInLimitMask]
        return inLimitJointValue

    def get_trajectory(path, time, jointSpeedLimit):
        pass


if __name__ == "__main__":
    from icecream import ic
    from datasave.joint_value.experiment_paper import Experiment2DArm
    from datasave.joint_value.pre_record_value import SinglePose

    qA = Experiment2DArm.PoseSingle.xApp
    a = JointProcess.find_shifted_value(qA)
    ic(a)
    ic(a.shape)

    # q = SinglePose.Pose6.thetaApp
    # b = JointProcess.find_shifted_value(q)
    # ic(b)
    # ic(b.shape)