""" Newton Euler interative dynamic
Index naming:
- variable_i_i = variable i to i
- variable_ip1_i = variable i+1 to i
"""

import numpy as np


class NewtonEuler:

    def outward(T_i_ip1, omega_i_i, omegadot_i_i, v_i_i, vdot_i_i, thetadot_ip1, thetadotdot_ip1, p_i_ip1, pc_ip1_ip1, m_ip1, Ic_ip1_ip1):

        R_i_ip1 = np.array([[T_i_ip1[0, 0], T_i_ip1[0, 1], T_i_ip1[0, 2]], [T_i_ip1[1, 0], T_i_ip1[1, 1], T_i_ip1[1, 2]], [T_i_ip1[2, 0], T_i_ip1[2, 1], T_i_ip1[2, 2]]])
        R_ip1_i = R_i_ip1.T
        Z_ip1_ip1 = np.array([0, 0, 1]).reshape(3, 1)

        # omega
        omega_ip1_ip1 = R_ip1_i@omega_i_i + thetadot_ip1@Z_ip1_ip1

        # omega dot
        omegadot_ip1_ip1 = R_ip1_i @ omegadot_i_i \
                        + np.cross((R_ip1_i @ omega_i_i), (thetadot_ip1 @ Z_ip1_ip1), axis=0) \
                        + thetadotdot_ip1 @ Z_ip1_ip1

        # v
        v_ip1_ip1 = R_ip1_i @ (v_i_i + np.cross(omega_i_i, p_i_ip1, axis=0)) # rotary joint
        # v_ip1_ip1 = R_ip1_i @ (v_i_i + np.cross(omega_i_i, p_i_ip1, axis=0)) + ddot_ip1 @ Z_ip1_ip1 # prismatic joint

        # v dot
        vdot_ip1_ip1 = R_ip1_i @ (np.cross(omegadot_i_i, p_i_ip1, axis=0) \
                        + np.cross(omega_i_i, np.cross(omega_i_i, p_i_ip1, axis=0), axis=0) \
                        + vdot_i_i)

        # v center
        vcdot_ip1_ip1 = np.cross(omegadot_ip1_ip1, pc_ip1_ip1, axis=0) \
                        + np.cross(omega_ip1_ip1, np.cross(omega_ip1_ip1, pc_ip1_ip1, axis=0), axis=0) \
                        + vdot_ip1_ip1

        # force
        F_ip1_ip1 = m_ip1 * vcdot_ip1_ip1

        # moment
        N_ip1_ip1 = Ic_ip1_ip1 @ omegadot_ip1_ip1 \
                        + np.cross(omega_ip1_ip1, (Ic_ip1_ip1 @ omega_ip1_ip1), axis=0)

        return omega_ip1_ip1, omegadot_ip1_ip1, v_ip1_ip1, vdot_ip1_ip1, vcdot_ip1_ip1, F_ip1_ip1, N_ip1_ip1

    def inward(T_i_ip1, f_ip1_ip1, F_i_i, n_ip1_ip1, N_i_i, pc_i_i, p_i_ip1):

        R_i_ip1 = np.array([[T_i_ip1[0, 0], T_i_ip1[0, 1], T_i_ip1[0, 2]], [T_i_ip1[1, 0], T_i_ip1[1, 1], T_i_ip1[1, 2]], [T_i_ip1[2, 0], T_i_ip1[2, 1], T_i_ip1[2, 2]]])
        z_i_i = np.array([0, 0, 1]).reshape(3, 1)

        # force per joint
        f_i_i = R_i_ip1@f_ip1_ip1 + F_i_i

        # moment per joint
        n_i_i = N_i_i + (R_i_ip1 @ n_ip1_ip1) \
                + np.cross(pc_i_i, F_i_i, axis=0) \
                + np.cross(p_i_ip1, (R_i_ip1 @ f_ip1_ip1), axis=0)

        # torque per joint
        t_i = n_i_i.T @ z_i_i

        return f_i_i, n_i_i, t_i
