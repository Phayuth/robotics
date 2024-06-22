import numpy as np
from itertools import product


class Utilities:

    def map_val(x, inMin, inMax, outMin, outMax):
        return (x - inMin) * (outMax - outMin) / (inMax - inMin) + outMin

    def wrap_to_pi(q):
        return (q + np.pi) % (2 * np.pi) - np.pi

    def find_shifted_value(q, filterOriginalq=False):  # -> Any:  # input must be of shape (n,1)
        shiftedComb = np.array(list(product([-2.0 * np.pi, 0.0, 2.0 * np.pi], repeat=q.shape[0]))).T
        shiftedJointValue = shiftedComb + q
        isInLimitCheck = np.logical_and(shiftedJointValue >= -2 * np.pi, shiftedJointValue <= 2 * np.pi)
        isInLimitMask = np.all(isInLimitCheck, axis=0)
        inLimitJointValue = shiftedJointValue[:, isInLimitMask]

        if filterOriginalq:
            exists = np.all(inLimitJointValue == q, axis=0)
            filterout = inLimitJointValue[:, ~exists]
            return filterout

        return inLimitJointValue

    def find_alt_config(q, configLimit, configConstrict=None, filterOriginalq=False):  # a better function than find_shifted_value
        """
        Find the alternative value of configuration in different quadrand.

        Parameters
        ----------
        q : array shape [state x 1]
            Original configuration value.
        configLimit : array shape [state x 2] with left and right limit. Ex: np.array([[-2 * np.pi, 2 * np.pi], [-np.pi, np.pi]])
            Physical limit of robot
        configConstrict : List, optional
            Constrict specific joint from finding alternative. Ex: the last joint of robot doesn't make any different when moving so we ignore them.
        filterOriginalq : bool, optional
            Filter out the original q value. Keep only the alternative value in array. by default False

        Returns
        -------
        array shape [state x alt.number]
            Alternative value
        """
        # possible config value
        q = Utilities.wrap_to_pi(q)  # transform to base quadrand first
        qShifted = q + np.array(list(product([-2.0 * np.pi, 0.0, 2.0 * np.pi], repeat=q.shape[0]))).T

        # eliminate with joint limit
        isInLimitMask = np.all((qShifted >= configLimit[:, 0, np.newaxis]) & (qShifted <= configLimit[:, 1, np.newaxis]), axis=0)
        qInLimit = qShifted[:, isInLimitMask]

        # joint constrict
        if configConstrict is not None:
            assert isinstance(configConstrict, list), "configConstrict must be in list format"
            assert len(configConstrict) == q.shape[0], "configConstrict length must be equal to state number"
            for i in range(len(configConstrict)):
                if configConstrict[i] is True:
                    qInLimit[i] = q[i]

        if filterOriginalq:
            exists = np.all(qInLimit == q, axis=0)
            filterout = qInLimit[:, ~exists]
            return filterout

        return qInLimit

    def sort_config(qs, qAlts):
        dist = np.linalg.norm(qAlts - qs, axis=0)
        return np.argsort(dist)


if __name__ == "__main__":
    import os
    import sys

    wd = os.path.abspath(os.getcwd())
    sys.path.append(str(wd))

    from icecream import ic
    from datasave.joint_value.experiment_paper import Experiment2DArm
    from datasave.joint_value.pre_record_value import SinglePose

    def find_alt():
        # 2 dof
        q2 = Experiment2DArm.PoseSingle.xApp
        a = Utilities.find_shifted_value(q2)
        ic(a)
        ic(a.shape)

        limt2 = np.array([[-2 * np.pi, 2 * np.pi], [-np.pi, np.pi]])
        gg = Utilities.find_alt_config(q2, limt2)
        ic(gg)

        # 3 dof
        limt3 = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])
        p = Utilities.find_alt_config(np.array([1, 1, 1]).reshape(3, 1), limt3)
        ic(p.shape)

        # 4 dof
        limt4 = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])
        i = Utilities.find_alt_config(np.array([1, 1, 1, 1]).reshape(4, 1), limt4)
        ic(i.shape)

        # 6 dof
        q6 = SinglePose.Pose6.thetaApp
        limt6 = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi], [-np.pi, np.pi], [-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])
        const = [False, False, False, False, False, False]
        u = Utilities.find_alt_config(q6, limt6, const, filterOriginalq=False)
        ic(u.shape)

    find_alt()

    def mapvalue():
        m = 2
        n = Utilities.map_val(m, 0, 5, 0, 100)
        ic(n)

        q = np.random.random((6, 1))
        w = Utilities.map_val(q, 0, 1, 0, 100)
        ic(w)
