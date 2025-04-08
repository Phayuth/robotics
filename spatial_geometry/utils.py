import numpy as np
from itertools import product


class Utils:

    def map_val(x, inMin, inMax, outMin, outMax):
        """
        >>> m = 2
        >>> n = Utils.map_val(m, 0, 5, 0, 100)
        >>> n : 40.0

        >>> q = np.random.random((6, 1))
        >>> w = Utils.map_val(q, 0, 1, 0, 100)
        >>> w : [[ 4.21994177], [88.67016363], [64.28742299], [61.37314135], [18.75513219], [35.97633713]]

        """
        return (x - inMin) * (outMax - outMin) / (inMax - inMin) + outMin

    def wrap_to_pi(q):
        return (q + np.pi) % (2 * np.pi) - np.pi

    def find_alt_config(q, configLimit, configConstrict=None, filterOriginalq=False):  # a better function than find_shifted_value
        """
        Find the alternative configuration.
        configConstrict : Constrict specific joint from finding alternative. Ex: the last joint of robot doesn't make any different when moving so we ignore them.
        filterOriginalq : Filter out the original q value. Keep only the alternative value in array. by default False

        # 2 DOF
        >>> q2 = Experiment2DArm.PoseSingle.xApp
        >>> limt2 = np.array([[-2 * np.pi, 2 * np.pi], [-np.pi, np.pi]])
        >>> gg = Utils.find_alt_config(q2, limt2)
        >>> gg : [[-4.8,  1.5], [ 0.2,  0.2]]

        # 6 DOF
        >>> q6 = SinglePose.Pose6.thetaApp
        >>> limt6 = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi], [-np.pi, np.pi], [-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])
        >>> const = [False, False, False, False, False, False]
        >>> u = Utils.find_alt_config(q6, limt6, const, filterOriginalq=False)
        >>> u : [[-1.9, -1.9, -1.9, -1.9, -1.9, -1.9, -1.9, -1.9, -1.9, -1.9, ...,  4.4,  4.4,  4.4,  4.4,  4.4,  4.4,  4.4,  4.4,  4.4,  4.4,  4.4,  4.4],
                 [-1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1,  5.2,  5.2, ..., -1.1, -1.1, -1.1, -1.1,  5.2,  5.2,  5.2,  5.2,  5.2,  5.2,  5.2,  5.2],
                 [-2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, ..., -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1, -2.1],
                 [-6.1, -6.1, -6.1, -6.1,  0.2,  0.2,  0.2,  0.2, -6.1, -6.1, ...,  0.2,  0.2,  0.2,  0.2, -6.1, -6.1, -6.1, -6.1,  0.2,  0.2,  0.2,  0.2],
                 [-4.7, -4.7,  1.6,  1.6, -4.7, -4.7,  1.6,  1.6, -4.7, -4.7, ..., -4.7, -4.7,  1.6,  1.6, -4.7, -4.7,  1.6,  1.6, -4.7, -4.7,  1.6,  1.6],
                 [-0.1,  6.2, -0.1,  6.2, -0.1,  6.2, -0.1,  6.2, -0.1,  6.2, ..., -0.1,  6.2, -0.1,  6.2, -0.1,  6.2, -0.1,  6.2, -0.1,  6.2, -0.1,  6.2]]

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
        """
        >>> qa = np.array([3.1, 0.0]).reshape(2, 1)
        >>> qb = np.array([-3.1, 0.0]).reshape(2, 1)
        >>> aa = Utils.minimum_dist_torus(qa, qb)
        >>> aa : 0.08318530717958605
        """
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

    def interpolate_so2(qfrom, qto, t):
        """
        >>> qfrom = -2.0
        >>> qto = -3.0
        >>> t = 0.5

        >>> qnew = interpolate_so2(qfrom, qto, t)
        >>> qnew : -2.5
        """
        diff = qto - qfrom

        if abs(diff) <= np.pi:
            qnew = qfrom + diff * t

        else:
            if diff > 0.0:
                diff = 2.0 * np.pi - diff
            else:
                diff = -2.0 * np.pi - diff
            qnew = qfrom - diff * t

            # input states are within bounds, so the following check is sufficient
            # if we want to unwrap the value. maybe be we dont want to wrap it back
            if qnew > np.pi:
                qnew -= 2 * np.pi
            elif qnew < -np.pi:
                qnew += 2 * np.pi

        return qnew

    def interpolate_so2_nd(qfrom, qto, t):
        """
        >>> qfrom = np.array([[-3.0], [0.0]])
        >>> qto = np.array([[3.0], [0.0]])
        >>> t = np.array([0.5, 0.5])
        >>> qnew = interpolate_so2_nd(qfrom, qto, t)
        >>> qnew : [[-3.14159265], [ 0.        ]]
        """
        qnew = np.zeros_like(qfrom)
        for i in range(qfrom.shape[0]):
            qnew[i] = Utils.interpolate_so2(qfrom[i], qto[i], t[i])
        return qnew

    def unwrap_so2(qfrom, qto):
        """
        >>> qfrom = -3.0
        >>> qto = 3.0

        >>> qnew = unwrap_so2(qfrom, qto)
        >>> qnew : -3.2831853071795862
        """
        # if we want to unwrap the value. maybe be we dont want to wrap it back
        t = 1.0  # it have to be 1.0
        diff = qto - qfrom

        if abs(diff) <= np.pi:
            qnew = qfrom + diff * t

        else:
            if diff > 0.0:
                diff = 2.0 * np.pi - diff
            else:
                diff = -2.0 * np.pi - diff
            qnew = qfrom - diff * t

        return qnew

    def unwrap_so2_path1d(path1d):
        """
        >>> path1d = np.array([-2.0, -3.0, 3.0, 2.0])

        >>> pathunwrap = unwrap_so2_path1d(path1d)
        >>> pathunwrap : [-2.0, -3.0, -3.2831853071795862, -4.283185307179586]
        """
        pathunwrap = [path1d[0]]
        for i in range(path1d.shape[0] - 1):
            qnewuw = Utils.unwrap_so2(pathunwrap[i], path1d[i + 1])
            pathunwrap.append(qnewuw)
        return np.array(pathunwrap)

    def unwrap_so2_path(path):
        """
        >>> path = np.array([[-2.0, -3.0, 3.0, 2.0], [0.0, 0.0, 0.0, 0.0]])

        >>> pathunwrap = unwrap_so2_path(path)
        >>> pathunwrap : [[-2.0, -3.0, -3.2831853071795862, -4.283185307179586], [0.0, 0.0, 0.0, 0.0]]

        >>> plt.plot(path[0], path[1], "bo")
        >>> plt.show()
        """
        pathunwrap = np.zeros_like(path)
        for i in range(path.shape[0]):
            pathunwrap[i] = Utils.unwrap_so2_path1d(path[i])
        return pathunwrap
