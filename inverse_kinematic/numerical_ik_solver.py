import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

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


if __name__ == "__main__":
    from inverse_kinematic.algorithm_solver import IKJacobianInverseSolver, IKJacobianPseudoInverseSolver, IKJacobianTransposeSolver, IKDampedLeastSquareSolver
    from robot.planar_rr import PlanarRR

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