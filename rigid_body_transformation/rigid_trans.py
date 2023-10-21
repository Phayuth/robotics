import numpy as np


class RigidBodyTransformation:
    
    def hrx(theta):
        return np.array([[1,             0,              0,  0],
                         [0, np.cos(theta), -np.sin(theta),  0],
                         [0, np.sin(theta),  np.cos(theta),  0],
                         [0,             0,              0,  1]])

    def hry(theta):
        return np.array([[np.cos(theta),  0,  np.sin(theta),  0],
                         [            0,  1,              0,  0],
                         [-np.sin(theta), 0,  np.cos(theta),  0],
                         [0,              0,              0,  1]])

    def hrz(theta):
        return np.array([[np.cos(theta), -np.sin(theta),  0,  0],
                         [np.sin(theta),  np.cos(theta),  0,  0],
                         [            0,              0,  1,  0],
                         [            0,              0,  0,  1]])

    def ht(x, y, z):
        return np.array([[            1,              0,  0,  x],
                         [            0,              1,  0,  y],
                         [            0,              0,  1,  z],
                         [            0,              0,  0,  1]])

    def h_inverse(H):
        R = H[0:3, 0:3]
        P = H[0:3, 3, np.newaxis]

        upper = np.hstack((R.T, -R.T @ P))
        lower = np.array([[0, 0, 0, 1]])
        return np.vstack((upper, lower))

    def rot2d(theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta),  np.cos(theta)]])

    def rotx(theta):
        return np.array([[1,             0,              0],
                         [0, np.cos(theta), -np.sin(theta)],
                         [0, np.sin(theta),  np.cos(theta)]])

    def roty(theta):
        return np.array([[np.cos(theta),  0,  np.sin(theta)],
                         [            0,  1,              0],
                         [-np.sin(theta), 0,  np.cos(theta)]])

    def rotz(theta):
        return np.array([[np.cos(theta), -np.sin(theta),  0],
                         [np.sin(theta),  np.cos(theta),  0],
                         [            0,              0,  1]])

    def dh_transformation(theta, alpha, d, a): # original dh method
        return np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                       [  np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                       [              0,                np.sin(alpha),                np.cos(alpha),               d],
                       [              0,                            0,                            0,               1]])

    def dh_transformation_mod(theta, alpha, d, a): # modified dh method from craig
        return np.array([[              np.cos(theta),              -np.sin(theta),              0,                a],
                         [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha), -np.sin(alpha)*d],
                         [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha),  np.cos(alpha),  np.cos(alpha)*d],
                         [                          0,                           0,              0,                1]])

    def post_multiply(*argv):
        """
        sequence of rotation
        for concurrent frame rotation, we post multiply of rotation matrix.
            theta = np.deg2rad(90)
            p1 = np.array([[1],[0],[0]])
            p2 = rotation_3d_y_axis(theta) @ rotation_3d_z_axis(theta) @ p1

        """
        rotation_list = []
        for arg in argv:
            rotation_list.append(arg)

        rot = rotation_list[0]
        for i in range(1, len(rotation_list)):
            rot = rot @ rotation_list[i]

        return rot

    def pre_multiply(*argv):
        """
        sequence of rotation
        fixed frame rotation, we pre multiply of rotation matrix.
            theta = np.deg2rad(90)
            p1 = np.array([[1],[0],[0]])
            p2 = rotation_3d_z_axis(theta) @ rotation_3d_y_axis(theta) @ p1

        """
        rotation_list = []
        for arg in argv:
            rotation_list.append(arg)

        rot = rotation_list[-1]
        for i in range(-2, -len(rotation_list), -1):
            rot = rotation_list[i] @ rot

        return rot

    def rot_fix_ang(seq, angle_seq):
        """rotation in fixed coordinate system. front multiply.
        
        Args:
            seq (str): sequence of rotation. ex: xyz, zyx, xyx, ...
            angle_seq (np.ndarray): the input of angle_seq sequence must be (gamma(x), beta(y), alpha(z))
        """
        if len(seq) != 3:
            return None
        else:
            rotation = np.identity(3)
            for index, value in enumerate(seq):
                if value == 'x':
                    rotation = RigidBodyTransformation.rotx(angle_seq[index,0]) @ rotation
                elif value == 'y':
                    rotation = RigidBodyTransformation.roty(angle_seq[index,0]) @ rotation
                elif value == 'z':
                    rotation = RigidBodyTransformation.rotz(angle_seq[index,0]) @ rotation

            return rotation

    def rot_euler(seq, angle_seq):
        """rotation in euler coordinate system. back multiply.
        not correct yet
        Args:
            seq (str): sequence of rotation. ex: xyz, zyx, xyx, ...
            angle_seq (np.ndarray): the input of angle_seq sequence must be (gamma(x), beta(y), alpha(z))
        """
        if len(seq) != 3:
            return None
        else:
            rotation = np.identity(3)
            for index, value in enumerate(seq): # reversed oder
                if value == 'x':
                    rotation = rotation @ RigidBodyTransformation.rotx(angle_seq[-index-1,0])
                elif value == 'y':
                    rotation = rotation @ RigidBodyTransformation.roty(angle_seq[-index-1,0])
                elif value == 'z':
                    rotation = rotation @ RigidBodyTransformation.rotz(angle_seq[-index-1,0])

            return rotation

    def conv_fixang_to_rotvec(rotmat):
        r11 = rotmat[0,0]
        r12 = rotmat[0,1]
        r21 = rotmat[1,0]
        r22 = rotmat[1,1]
        r31 = rotmat[2,0]
        r32 = rotmat[2,1]
        r33 = rotmat[2,2]

        beta = np.arctan2(-r31,np.sqrt(r11**2 + r21**2))

        if abs(beta - np.pi/2) > 1e-6 and abs(beta + np.pi/2) > 1e-6:
            beta = np.arctan2(-r31,np.sqrt(r11**2 + r21**2))
            alpha = np.arctan2(r21/np.cos(beta), r11/np.cos(beta))
            gamma = np.arctan2(r32/np.cos(beta), r33/np.cos(beta))

        elif beta == np.pi/2: # gimal lock, by convention we choose alpha = 0
            beta = np.pi/2
            alpha = 0
            gamma = np.arctan2(r12, r22)

        elif beta == -np.pi/2: # gimbal lock, by convention we choose alpha = 0
            beta = -np.pi/2
            alpha = 0
            gamma = -np.arctan2(r12, r22)

        return np.array([gamma, beta, alpha]).reshape(3,1)

    def conv_euler_to_rotvec(rotmat):
        r11 = rotmat[0,0]
        r12 = rotmat[0,1]
        r13 = rotmat[0,2]
        r23 = rotmat[1,2]
        r31 = rotmat[2,0]
        r32 = rotmat[2,1]
        r33 = rotmat[2,2]

        beta = np.arctan2(np.sqrt(r31**2 + r32**2), r33)

        if 0 < beta < np.pi:
            beta = np.arctan2(np.sqrt(r31**2 + r32**2), r33)
            alpha = np.arctan2(r23/np.sin(beta), r13/np.sin(beta))
            gamma = np.arctan2(r32/np.sin(beta), -r31/np.sin(beta))

        elif beta == 0: # gimbal lock, by convention we choose alpha = 0
            beta = 0
            alpha = 0
            gamma = np.arctan2(-r12, r11)

        elif beta == np.pi: # gimbal lock, by convention we choose alpha = 0
            beta = np.pi
            alpha = 0
            gamma = np.arctan2(r12, -r11)

        return np.array([gamma, beta, alpha]).reshape(3,1)

    def conv_axang_to_quat(theta, n):
        # quaternion = q0 + iq1 + jq2 +kq3
        # is a rotation by theta about unit vector n = (nx,ny,nz)
        norm = np.linalg.norm(n) # calculate norm of vector
        nx = n[0,0]/norm # unit vector in x direction
        ny = n[1,0]/norm # unit vector in y direction
        nz = n[2,0]/norm # unit vector in z direction

        q0 = np.cos(theta/2)
        q1 = nx*np.sin(theta/2)
        q2 = ny*np.sin(theta/2)
        q3 = nz*np.sin(theta/2)

        return np.array([q0,q1,q2,q3]).reshape(4,1)

    def conv_quat_to_axang(q):
        q0 = q[0,0]
        q1 = q[1,0]
        q2 = q[2,0]
        q3 = q[3,0]

        theta = np.arccos(q0)*2
        nx = q1/np.sin(theta/2)
        ny = q2/np.sin(theta/2)
        nz = q3/np.sin(theta/2)

        return theta, np.array([nx,ny,nz]).reshape(3,1)

    def conv_rotmat_to_axang(rotmat):
        r11 = rotmat[0,0]
        r12 = rotmat[0,1]
        r13 = rotmat[0,2]
        r21 = rotmat[1,0]
        r22 = rotmat[1,1]
        r23 = rotmat[1,2]
        r31 = rotmat[2,0]
        r32 = rotmat[2,1]
        r33 = rotmat[2,2]

        theta = np.arccos((r11 + r22 + r33 - 1)/2)
        k = (1/(2*np.sin(theta))) * np.array([r32 - r23, r13 - r31, r21 - r12]).reshape(3,1)

        return theta, k

    def conv_axang_to_rotmat(theta, k):
        k = k/np.linalg.norm(k)
        kx = k[0,0]
        ky = k[1,0]
        kz = k[2,0]
        vt = 1 - np.cos(theta)

        rotmat = np.array([[   kx*kx*vt + np.cos(theta), kx*ky*vt - kz*np.sin(theta), kx*kz*vt + ky*np.sin(theta)],
                           [kx*ky*vt + kz*np.sin(theta),    ky*ky*vt + np.cos(theta), ky*kz*vt - kx*np.sin(theta)],
                           [kx*kz*vt - ky*np.sin(theta), ky*kz*vt + kx*np.sin(theta),    kz*kz*vt + np.cos(theta)]])

        return rotmat

    def conv_quat_to_rotmat(q):
        q = q / np.linalg.norm(q)
        q0 = q[0,0]
        q1 = q[1,0]
        q2 = q[2,0]
        q3 = q[3,0]

        rotmat = np.array([[1 - 2*q2**2 - 2*q3**2,     2*q1*q2 - 2*q0*q3,     2*q1*q3 + 2*q0*q2],
                           [    2*q1*q2 + 2*q0*q3, 1 - 2*q1**2 - 2*q3**2,     2*q2*q3 - 2*q0*q1],
                           [    2*q1*q3 - 2*q0*q2,     2*q2*q3 + 2*q0*q1, 1 - 2*q1**2 - 2*q2**2]])

        return rotmat

    def conv_rotmat_to_quat(rotmat):
        r11, r12, r13 = rotmat[0, 0], rotmat[0, 1], rotmat[0, 2]
        r21, r22, r23 = rotmat[1, 0], rotmat[1, 1], rotmat[1, 2]
        r31, r32, r33 = rotmat[2, 0], rotmat[2, 1], rotmat[2, 2]

        q0 = np.sqrt(1 + r11 + r22 + r33) / 2
        q1 = (r32 - r23) / (4 * q0)
        q2 = (r13 - r31) / (4 * q0)
        q3 = (r21 - r12) / (4 * q0)

        return np.array([q0, q1, q2, q3]).reshape(4,1)

    def vec_to_skew(x):
        return np.array([[      0,  -x[2,0],  x[1,0]],
                         [ x[2,0],        0, -x[0,0]],
                         [-x[1,0],   x[0,0],       0]])

    def basis_vec(ax):
        if ax == 'x':
            return np.array([[1], [0], [0]])
        elif ax == 'y':
            return np.array([[0], [1], [0]])
        elif ax == 'z':
            return np.array([[0], [0], [1]])
        
    def derivative_rot(rotAx, theta): # Derivative of rotation matrix about x = skew matrix of x basis vector @ Rotation Matrix x
        if rotAx == 'x':
            return RigidBodyTransformation.vec_to_skew(RigidBodyTransformation.basis_vec(rotAx)) @ RigidBodyTransformation.rotx(theta)
        elif rotAx == 'y':
            return RigidBodyTransformation.vec_to_skew(RigidBodyTransformation.basis_vec(rotAx)) @ RigidBodyTransformation.roty(theta)
        elif rotAx == 'z':
            return RigidBodyTransformation.vec_to_skew(RigidBodyTransformation.basis_vec(rotAx)) @ RigidBodyTransformation.rotz(theta)

    def derivative_rot_velocity(rotAx, theta, thetaDot): # with omega, omega = (i vector basis) @ thetaDot
        if rotAx == 'x':
            return RigidBodyTransformation.vec_to_skew(thetaDot*RigidBodyTransformation.basis_vec(rotAx)) @ RigidBodyTransformation.rotx(theta)
        elif rotAx == 'y':
            return RigidBodyTransformation.vec_to_skew(thetaDot*RigidBodyTransformation.basis_vec(rotAx)) @ RigidBodyTransformation.roty(theta)
        elif rotAx == 'z':
            return RigidBodyTransformation.vec_to_skew(thetaDot*RigidBodyTransformation.basis_vec(rotAx)) @ RigidBodyTransformation.rotz(theta)


class PlotTransformation:

    def plot_frame_2d(rot2d, translation, ax, plt_basis=False):

        r1 = np.array([[0],[0]])
        r2 = np.array([[1],[0]])
        r4 = np.array([[0],[1]])

        dx = translation[0,0]
        dy = translation[1,0]

        d1 = np.array([[dx],[dy]])
        d2 = np.array([[dx],[dy]])
        d4 = np.array([[dx],[dy]])

        # rnew = Rotation @ rold + d
        r1new = rot2d @ r1 + d1
        r2new = rot2d @ r2 + d2
        r4new = rot2d @ r4 + d4

        ax.set_aspect('equal')
        if plt_basis:
            # plot basic axis
            ax.axvline(x=0, c="black")
            ax.axhline(y=0, c="black")

        # plot frame
        ax.plot([r1new[0,0], r2new[0,0]],[r1new[1,0], r2new[1,0]],"blue", linewidth=4) # x axis
        ax.plot([r1new[0,0], r4new[0,0]],[r1new[1,0], r4new[1,0]],"red", linewidth=4)  # y axis

 
    def plot_frame_3d(H, ax, plt_basis=False): # input 4x4 transform matrix. use ax = plt.axes(projection='3d')
        rotation = H[0:3, 0:3]
        d = H[0:3, 3, np.newaxis]

        r1 = np.array([[1],[0],[0]])
        r2 = np.array([[0],[1],[0]])
        r3 = np.array([[0],[0],[1]])
        r4 = np.array([[0],[0],[0]])

        r1new = rotation @ r1 + d
        r2new = rotation @ r2 + d
        r3new = rotation @ r3 + d
        r4new = rotation @ r4 + d

        if plt_basis:
            # plot basic axis
            ax.plot3D([0, 2], [0, 0], [0, 0], 'red', linewidth=4)
            ax.plot3D([0, 0], [0, 2], [0, 0], 'purple', linewidth=4)
            ax.plot3D([0, 0], [0, 0], [0, 2], 'green', linewidth=4)

        # plot frame
        ax.plot3D([r4new[0,0], r1new[0,0]], [r4new[1,0], r1new[1,0]], [r4new[2,0], r1new[2,0]], 'gray', linewidth=4, label="gray is x")
        ax.plot3D([r4new[0,0], r2new[0,0]], [r4new[1,0], r2new[1,0]], [r4new[2,0], r2new[2,0]], 'blue', linewidth=4, label="blue is y")
        ax.plot3D([r4new[0,0], r3new[0,0]], [r4new[1,0], r3new[1,0]], [r4new[2,0], r3new[2,0]], 'yellow', linewidth=4, label="yellow is z")
        ax.legend()