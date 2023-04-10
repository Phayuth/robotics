""" Inverse Kinematic based on numerical root finding method.
- Method : Damped leastsquare / Damped leastsquare clampmag
- Return : 1 Possible Theta

"""

import numpy as np
from clampMag import clampMag


class ik_damped_leastsquare:

    def __init__(self, max_iteration, robot_class, ls_damp_cte):
        self.max_iter = max_iteration  # for when it can't reach desired pose
        self.robot = robot_class
        self.damp_cte = ls_damp_cte
        self.theta_history = np.array([[]])

    def damped_least_square(self, theta_current, x_desired):
        x_current = self.robot.forward_kinematic(theta_current)
        e = x_desired - x_current
        i = 0

        while np.linalg.norm(e) > 0.001 and i < self.max_iter:

            x_current = self.robot.forward_kinematic(theta_current)
            e = x_desired - x_current
            Jac = self.robot.jacobian(theta_current)
            delta_theta = np.transpose(Jac) @ np.linalg.inv(Jac @ np.transpose(Jac) + np.identity(2) * (self.damp_cte**2)) @ e
            theta_current = theta_current + delta_theta
            i += 1

        return theta_current

    def damped_least_square_ClampMag(self, theta_current, x_desired, Dmax):
        x_current = self.robot.forward_kinematic(theta_current)
        e = x_desired - x_current
        e = clampMag(e, Dmax)
        i = 0

        while np.linalg.norm(e) > 0.001 and i < self.max_iter:

            x_current = self.robot.forward_kinematic(theta_current)
            e = x_desired - x_current
            e = clampMag(e, Dmax)
            Jac = self.robot.jacobian(theta_current)
            delta_theta = np.transpose(Jac) @ np.linalg.inv(Jac @ np.transpose(Jac) + np.identity(2) * (self.damp_cte**2)) @ e
            theta_current = theta_current + delta_theta
            i += 1

        return theta_current
