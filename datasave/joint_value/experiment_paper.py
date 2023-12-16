import numpy as np


class Experiment2DArm:

    class PoseSingle:
        xStart = np.array([0, 0]).reshape(2, 1)
        xGoal = np.array([np.pi / 2, 0]).reshape(2, 1)
        xApp = np.array([np.pi / 2 - 0.1, 0.2]).reshape(2, 1)

    class PoseMulti:
        xStart = np.array([0, 0]).reshape(2, 1)
        xApp = [np.array([np.pi / 2 - 0.1, 0.2]).reshape(2, 1),
                np.array([1.45, -0.191]).reshape(2, 1),
                np.array([1.73, -0.160]).reshape(2, 1)]
        xGoal = [np.array([np.pi / 2, 0]).reshape(2, 1),
                 np.array([np.pi / 2, 0]).reshape(2, 1),
                 np.array([np.pi / 2, 0]).reshape(2, 1)]


class ICRABarnMap:

    class PoseSingle:
        xStart = np.array([[-2.70], [2.20]])
        xApp = np.array([[2.30], [-1.85]])
        xGoal = np.array([[2.30], [-2.30]])

    class PoseMulti: # one pre-grasp per one grasp pair
        xStart = np.array([[-2.70], [2.20]])
        xApp = [np.array([[2.30], [-1.85]]),
                np.array([[2.50], [-0.37]]),
                np.array([[1.95], [0.07]])]
        xGoal = [np.array([[2.30], [-2.30]]),
                 np.array([[2.50], [-0.85]]),
                 np.array([[1.80], [-0.27]])]

    class PoseMulti2: # one pre-grasp per one grasp pair
        xStart = np.array([[-2.70], [2.20]])
        xApp = [np.array([[2.30], [-1.85]]),
                np.array([[2.50], [-0.37]]),
                np.array([[0.05], [-2.33]])]
        xGoal = [np.array([[2.30], [-2.30]]),
                 np.array([[2.50], [-0.85]]),
                 np.array([[0.47], [-2.46]])]

    class PoseMulti3: # 3 pre-grasp per one grasp, [used in paper]
        xStart = np.array([[-2.70], [2.20]])
        xApp = [np.array([[2.30], [-1.85]]),
                np.array([[2.50], [-0.37]]),
                np.array([[0.05], [-2.33]])]
        xGoal = [np.array([[2.30], [-2.30]]),
                 np.array([[2.30], [-2.30]]),
                 np.array([[2.30], [-2.30]])]


class URHarvesting:

    class PoseSingle1:
        xStart = np.deg2rad([-0.39, -5.96, -3.43, 6.08, 1.68, -3.43]).reshape(6, 1)
        xApp = np.deg2rad([-104.63, -57.30, -114.53, -4.63, 87.33, 0.41]).reshape(6, 1)
        xGoal = np.deg2rad([-104.63, -73.92, -106.12, 7.80, 87.33, 0.41]).reshape(6, 1)

    class PoseSingle2:
        xStart = np.deg2rad([-0.39, -5.96, -3.43, 6.08, 1.68, -3.43]).reshape(6, 1)
        xApp = np.deg2rad([-135.55, -70.37, -105.53, -4.65, 132.33, 0.41]).reshape(6, 1)
        xGoal = np.deg2rad([-119.10, -79.48, -97.33, -0.32, 116.59, 0.41]).reshape(6, 1)

    class PoseSingle3:
        xStart = np.deg2rad([-0.39, -5.96, -3.43, 6.08, 1.68, -3.43]).reshape(6, 1)
        xApp = np.deg2rad([-54.07, -58.65, -111.73, 0.29, 15.82, 0.41]).reshape(6, 1)
        xGoal = np.deg2rad([-79.55, -72.13, -102.78, -1.98, 38.08, 0.41]).reshape(6, 1)

    class PoseMulti:
        xStart = np.deg2rad([-0.39, -5.96, -3.43, 6.08, 1.68, -3.43]).reshape(6, 1)

        xApp = [np.deg2rad([-104.63, -57.30, -114.53, -4.63, 87.33, 0.41]).reshape(6, 1),
                np.deg2rad([-135.55, -70.37, -105.53, -4.65, 132.33, 0.41]).reshape(6, 1),
                np.deg2rad([-54.07, -58.65, -111.73, 0.29, 15.82, 0.41]).reshape(6, 1)]

        xGoal = [np.deg2rad([-104.63, -73.92, -106.12, 7.80, 87.33, 0.41]).reshape(6, 1),
                 np.deg2rad([-119.10, -79.48, -97.33, -0.32, 116.59, 0.41]).reshape(6, 1),
                 np.deg2rad([-79.55, -72.13, -102.78, -1.98, 38.08, 0.41]).reshape(6, 1)]


if __name__ == "__main__":
    q = ICRABarnMap.PoseSingle()
    qS = q.xStart
    qA = q.xApp
    qG = q.xGoal