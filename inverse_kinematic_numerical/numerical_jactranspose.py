""" Inverse Kinematic based on numerical root finding method.
- Method : Transpose Jacobian
- Return : 1 Possible Theta

"""

import numpy as np


class ik_jacobian_transpose:

    def __init__(self, max_iteration, robot_class):
        self.max_iter = max_iteration  # for when it can't reach desired pose
        self.robot = robot_class
        self.theta_history = np.array([[]])

    def cal_alpha(self, e, Jac):
        return (np.dot(np.transpose(e), Jac @ np.transpose(Jac) @ e)) / (np.dot(np.transpose(Jac @ np.transpose(Jac) @ e), Jac @ np.transpose(Jac) @ e))

    def transpose_jac(self, theta_current, x_desired):
        x_current = self.robot.forward_kinematic(theta_current)
        e = x_desired - x_current
        i = 0

        while np.linalg.norm(e) > 0.001 and i < self.max_iter:

            x_current = self.robot.forward_kinematic(theta_current)
            e = x_desired - x_current
            Jac = self.robot.jacobian(theta_current)
            alpha = self.cal_alpha(e, Jac)
            theta_current = theta_current + alpha * np.transpose(Jac).dot(e)
            i += 1

        return theta_current
