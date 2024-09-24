import numpy as np
from itertools import product


class Utils:

    def map_val(x, inMin, inMax, outMin, outMax):
        return (x - inMin) * (outMax - outMin) / (inMax - inMin) + outMin

    def wrap_to_pi(q):
        return (q + np.pi) % (2 * np.pi) - np.pi

    def find_alt_config(q, configLimit, configConstrict=None, filterOriginalq=False):  # a better function than find_shifted_value
        """
        Find the alternative value of configuration in different quadrand.
        configConstrict : Constrict specific joint from finding alternative. Ex: the last joint of robot doesn't make any different when moving so we ignore them.
        filterOriginalq : Filter out the original q value. Keep only the alternative value in array. by default False
        """
        # possible config value
        qw = Utils.wrap_to_pi(q)  # transform to base quadrand first
        qShifted = qw + np.array(list(product([-2.0 * np.pi, 0.0, 2.0 * np.pi], repeat=qw.shape[0]))).T

        # eliminate with joint limit
        isInLimitMask = np.all((qShifted >= configLimit[:, 0, np.newaxis]) & (qShifted <= configLimit[:, 1, np.newaxis]), axis=0)
        qInLimit = qShifted[:, isInLimitMask]

        # joint constrict
        if configConstrict is not None:
            assert isinstance(configConstrict, list), "configConstrict must be in list format"
            assert len(configConstrict) == qw.shape[0], "configConstrict length must be equal to state number"
            for i in range(len(configConstrict)):
                if configConstrict[i] is True:
                    qInLimit[i] = qw[i]

        if filterOriginalq:
            exists = np.all(qInLimit == q, axis=0)
            filterout = qInLimit[:, ~exists]
            return filterout

        return qInLimit

    def sort_config(qs, qAlts):
        dist = np.linalg.norm(qAlts - qs, axis=0)
        return np.argsort(dist)

    def minimum_dist_torus(qa, qb):
        L = np.full_like(qa, 2 * np.pi)
        delta = np.abs(qa - qb)
        deltaw = L - delta
        deltat = np.min(np.hstack((delta, deltaw)), axis=1)
        return np.linalg.norm(deltat)

    def nearest_qb_to_qa(qa, qb, configLimit, ignoreOrginal=True):
        """
        if ignore_original there alway be torus path arround even the two point is close
        if not, then the original will be consider and if it has minimum distance there only 1 way to move.
        """
        Qb = Utils.find_alt_config(qb, configLimit, filterOriginalq=ignoreOrginal)
        di = Qb - qa
        n = np.linalg.norm(di, axis=0)
        minid = np.argmin(n)
        return Qb[:, minid, np.newaxis]


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
        limt2 = np.array([[-2 * np.pi, 2 * np.pi], [-np.pi, np.pi]])
        gg = Utils.find_alt_config(q2, limt2)
        ic(gg)

        # 6 dof
        q6 = SinglePose.Pose6.thetaApp
        limt6 = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi], [-np.pi, np.pi], [-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])
        const = [False, False, False, False, False, False]
        u = Utils.find_alt_config(q6, limt6, const, filterOriginalq=False)
        ic(u.shape)

    def mapvalue():
        m = 2
        n = Utils.map_val(m, 0, 5, 0, 100)
        ic(n)

        q = np.random.random((6, 1))
        w = Utils.map_val(q, 0, 1, 0, 100)
        ic(w)

    qa = np.array([3.1, 0.0]).reshape(2, 1)
    qb = np.array([-3.1, 0.0]).reshape(2, 1)
    aa = Utils.minimum_dist_torus(qa, qb)
    print(f"> aa: {aa}")
