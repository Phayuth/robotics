""" Inverse Kinematic based on numerical root finding method.
- Method : Selectively Damped leastsquare / Selectively Damped leastsquare clampmag
- Return : 1 Possible Theta

# not correct yet, fix later 

"""

import numpy as np
from clampMag import clampMagAbs


class ik_selectively_damped_leastsquare:

    def __init__(self, max_iteration, robot_class, gamma_max=np.pi / 4):
        self.max_iter = max_iteration  # for when it can't reach desired pose
        self.robot = robot_class
        self.gamma_max = gamma_max
        self.theta_history = np.array([[]])

    def rho11(self, theta1, theta2):
        eq1 = -self.l1 * np.sin(theta1)
        eq2 = self.l1 * np.cos(theta1)
        vector = np.array([[eq1], [eq2]])
        return np.linalg.norm(vector)

    def rho12(self, theta1, theta2):
        eq1 = 0
        eq2 = 0
        vector = np.array([[eq1], [eq2]])
        return np.linalg.norm(vector)

    def rho21(self, theta1, theta2):
        eq1 = -self.l1 * np.sin(theta1) - self.l2 * np.sin(theta1 + theta2)
        eq2 = self.l1 * np.cos(theta1) + self.l2 * np.cos(theta1 + theta2)
        vector = np.array([[eq1], [eq2]])
        return np.linalg.norm(vector)

    def rho22(self, theta1, theta2):
        eq1 = -self.l2 * np.sin(theta1 + theta2)
        eq2 = self.l2 * np.cos(theta1 + theta2)
        vector = np.array([[eq1], [eq2]])
        return np.linalg.norm(vector)

    def selectively_damped_least_square_fast(self, theta_current, x_desired):
        x_current = self.robot.forward_kinematic(theta_current)
        e = x_desired - x_current

        i = 0

        while np.linalg.norm(e) > 0.001 and i < self.max_iter:

            x_current = self.robot.forward_kinematic(theta_current)
            e = x_desired - x_current

            Jac = self.robot.jacobian(theta_current)
            U, D, VT = np.linalg.svd(Jac)
            V = np.transpose(VT)
            v1 = np.array([[V[0, 0]], [V[1, 0]]])
            v2 = np.array([[V[0, 1]], [V[1, 1]]])
            u1 = np.array([[U[0, 0]], [U[1, 0]]])
            u2 = np.array([[U[0, 1]], [U[1, 1]]])
            alpha1 = np.dot(np.transpose(e), u1)
            alpha2 = np.dot(np.transpose(e), u2)

            tau1 = 0.2
            tau2 = 0.2

            delta_theta = alpha1 * tau1 * v1 + alpha2 * tau2 * v2

            theta_current = theta_current + delta_theta
            i += 1
        return theta_current

    def selectively_damped_least_square(self, theta_current, x_desired):
        x_current = self.robot.forward_kinematic(theta_current)
        e = x_desired - x_current

        i = 0

        while np.linalg.norm(e) > 0.001 and i < self.max_iter:  # norm = sqrt(x^2 + y^2)

            x_current = self.robot.forward_kinematic(theta_current)
            e = x_desired - x_current

            Jac = self.robot.jacobian(theta_current)
            U, D, VT = np.linalg.svd(Jac)
            V = np.transpose(VT)
            v1 = np.array([[V[0, 0]], [V[1, 0]]])
            v2 = np.array([[V[0, 1]], [V[1, 1]]])

            u1 = np.array([[U[0, 0]], [U[1, 0]]])
            u2 = np.array([[U[0, 1]], [U[1, 1]]])

            alpha1 = np.dot(np.transpose(e), u1)
            alpha2 = np.dot(np.transpose(e), u2)

            v11 = V[0, 0]
            v12 = V[0, 1]
            v21 = V[1, 0]
            v22 = V[1, 1]

            sigma1 = D[0]
            sigma2 = D[1]

            M11 = (1 / sigma1) * (np.abs(v11) * self.rho11(theta_current[0, 0], theta_current[1, 0]) + np.abs(v21) * self.rho12(theta_current[0, 0], theta_current[1, 0]))
            M12 = (1 / sigma1) * (np.abs(v11) * self.rho21(theta_current[0, 0], theta_current[1, 0]) + np.abs(v21) * self.rho22(theta_current[0, 0], theta_current[1, 0]))

            M21 = (1 / sigma2) * (np.abs(v12) * self.rho11(theta_current[0, 0], theta_current[1, 0]) + np.abs(v22) * self.rho12(theta_current[0, 0], theta_current[1, 0]))
            M22 = (1 / sigma2) * (np.abs(v12) * self.rho21(theta_current[0, 0], theta_current[1, 0]) + np.abs(v22) * self.rho22(theta_current[0, 0], theta_current[1, 0]))

            M1 = M11 + M12
            M2 = M21 + M22

            N1 = np.sum(np.abs(u1))
            N2 = np.sum(np.abs(u2))

            gama1 = np.minimum(1, N1 / M1) * self.gama_max
            gama2 = np.minimum(1, N2 / M2) * self.gama_max

            c1 = (1 / sigma1) * alpha1 * v1
            c2 = (1 / sigma2) * alpha2 * v2

            phi1 = clampMagAbs(c1, gama1)
            phi2 = clampMagAbs(c2, gama2)

            delta_theta = clampMagAbs(phi1 + phi2, self.gama_max)

            theta_current = theta_current + delta_theta
            i += 1

        return theta_current
