import numpy as np


class NumericalInverseKinematic:

    def __init__(self, robotClass, maxIteration, epsilon, solver, *solverParameters) -> None:
        self.robotClass = robotClass
        self.solver = solver(self, *solverParameters)
        self.maxIteration = maxIteration
        self.epsilon = epsilon

        self.qCurrent = None
        self.desiredTaskPose = None

    def set_current_joint_value(self, q):
        self.qCurrent = q

    def get_current_task_pose(self):
        return self.robotClass.forward_kinematic(self.qCurrent)

    def get_jacobian(self):
        return self.robotClass.jacobian(self.qCurrent)

    def set_desired_task_pose(self, x):
        self.desiredTaskPose = x

    def get_error(self):
        return self.desiredTaskPose - self.get_current_task_pose()

    def get_error_norm(self):
        return np.linalg.norm(self.get_error())

    def solve(self):
        for itera in range(self.maxIteration):
            if self.get_error_norm() > self.epsilon:
                self.qCurrent = self.qCurrent + self.solver.update_term()

        return self.qCurrent


class IKJacobianInverseSolver:  # THIS METHOD OF IK IS NOT WORKING WHEN THE ARM IS AT SINGULARITY

    def __init__(self, baseClass, clampMagConstant=None) -> None:
        self.baseClass = baseClass
        self.clampMagConstant = clampMagConstant

    def update_term(self):
        e = self.baseClass.get_error()
        if self.clampMagConstant:
            e = clampMag(e, self.clampMagConstant)
        Jac = self.baseClass.get_jacobian()
        return np.linalg.inv(Jac).dot(e)


class IKJacobianPseudoInverseSolver:

    def __init__(self, baseClass) -> None:
        self.baseClass = baseClass

    def update_term(self):
        e = self.baseClass.get_error()
        Jac = self.baseClass.get_jacobian()
        return np.linalg.pinv(Jac).dot(e)


class IKJacobianTransposeSolver:

    def __init__(self, baseClass) -> None:
        self.baseClass = baseClass

    def cal_alpha(self, e, Jac):
        JacT = np.transpose(Jac)
        JacJacTe = Jac @ JacT @ e
        return (np.dot(np.transpose(e), JacJacTe)) / (np.dot(np.transpose(JacJacTe), JacJacTe))

    def update_term(self):
        e = self.baseClass.get_error()
        Jac = self.baseClass.get_jacobian()
        alpha = self.cal_alpha(e, Jac)
        return alpha * np.transpose(Jac).dot(e)


class IKDampedLeastSquareSolver:

    def __init__(self, baseClass, dampConstant, clampMagConstant=None) -> None:
        self.baseClass = baseClass
        self.dampConstant = dampConstant
        self.clampMagConstant = clampMagConstant

    def update_term(self):
        e = self.baseClass.get_error()
        if self.clampMagConstant:
            e = clampMag(e, self.clampMagConstant)
        Jac = self.baseClass.get_jacobian()
        JacT = np.transpose(Jac)
        return JacT @ np.linalg.inv(Jac@JacT + np.identity(2) * (self.dampConstant**2)) @ e


# class IKSelectivelyDampedLeastSquareSolver:

#     def __init__(self, baseClass, dampConstant, clampMagConstant=None) -> None:
#         self.baseClass = baseClass
#         self.dampConstant = dampConstant
#         self.clampMagConstant = clampMagConstant
#         self.gammaMax = np.pi / 4  #  the maximum permissible change in any joint angle in a single step

#     def update_term(self):  # NOT CORRECT YET
#         e = self.baseClass.get_error()
#         Jac = self.baseClass.get_jacobian()
#         U, Sigma, VT = np.linalg.svd(Jac)
#         V = VT.T
#         Alpha = U.T @ e
#         Gamma = np.minimum(1, N / M) * self.gammaMax
#         Phi = clampMagAbs()
#         return clampMagAbs(np.sum(Phi), self.gammaMax)


def clampMag(w, d):
    if euclideanNormW := np.linalg.norm(w) <= d:
        return w
    else:
        return d * (w/euclideanNormW)


def clampMagAbs(w, d):
    if oneNormW := np.max(abs(w)) <= d:
        return w
    else:
        return d * (w/oneNormW)


if __name__ == "__main__":
    import os
    import sys

    wd = os.path.abspath(os.getcwd())
    sys.path.append(str(wd))

    from robot.nonmobile.planar_rr import PlanarRR

    # Create Robot Class
    robot = PlanarRR()

    # Create Desired Pose and Current data
    theta = np.array([[0.], [0.]])  # vector 2x1
    xDesired = np.array([[0.5], [0.5]])  # vector 2x1

    solver = NumericalInverseKinematic(robot, 100, 0.001, IKJacobianTransposeSolver)
    solver.set_current_joint_value(theta)
    solver.set_desired_task_pose(xDesired)
    qSolution = solver.solve()

    print(qSolution)
    robot.plot_arm(qSolution, plt_basis=True, plt_show=True)