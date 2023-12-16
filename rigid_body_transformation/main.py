import numpy as np
from rigid_trans import RigidBodyTransformation as rbt, PlotTransformation as pltt
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# homog transform
theta = np.pi / 4

p1 = np.array([[1], [1], [1], [1]])  # the fourth element MUST be equal to 1

H01 = rbt.ht(5, 0, 0)
H12 = rbt.ht(0, 5, 0)
H23 = rbt.hrz(theta)
H = H01 @ H12 @ H23
p0 = H @ p1

# plot frame
# 2D
rot = rbt.rot2d(theta)
tran = np.array([[1],[1]])
fig, ax = plt.subplots()
pltt.plot_frame_2d(rot, tran, ax, plt_basis=True)
plt.show()

# 3D
# gs = rbt.hrz(theta)
# plot_frame_3d(gs, plt_basis=True, plt_show=True)

# transformation
gs = rbt.hrx(theta=1)
gs_iv = rbt.hinverse(gs)
check = gs @ gs_iv  # return Identity Matrix

tac = np.array([[np.cos(np.deg2rad(90)), np.cos(np.deg2rad(120)), np.cos(np.deg2rad(30)), 3],
                [np.cos(np.deg2rad(90)), np.cos(np.deg2rad(30)), np.cos(np.deg2rad(60)), 0],
                [np.cos(np.deg2rad(180)), np.cos(np.deg2rad(90)),np.cos(np.deg2rad(90)), 2],
                [0, 0, 0, 1]])

tca = rbt.hinverse(tac)
tac_rotation_mat = tac[0:3, 0:3]
tca_rotation_mat = tca[0:3, 0:3]

tcb = np.array([[np.cos(np.deg2rad(90 + 90 - 36.9)), np.cos(np.deg2rad(90 + 36.9)), np.cos(np.deg2rad(90)), 3],
                [np.cos(np.deg2rad(90)), np.cos(np.deg2rad(90)), np.cos(np.deg2rad(0)), 2],
                [np.cos(np.deg2rad(90 + 36.9)), np.cos(np.deg2rad(36.9)), np.cos(np.deg2rad(90)), 0], [0, 0, 0, 1]])
tcb_rotation_mat = tcb[0:3, 0:3]

# fixed axis rotation
gamma = np.random.uniform(-np.pi, np.pi)
beta = np.random.uniform(-np.pi, np.pi)
alpha = np.random.uniform(-np.pi, np.pi)
gamma_beta_alpha = np.array([gamma, beta, alpha]).reshape(3, 1)

fixed_angle = rbt.rot_fix_ang('xyz', gamma_beta_alpha)
ang = rbt.conv_fixang_to_rotvec(fixed_angle)
fixed_angle_again = rbt.rot_fix_ang('xyz', ang)
print("==>> fixed_angle original: \n", fixed_angle)
print("==>> fixed_angle inverse problem: \n", fixed_angle_again)

# equivalent axis angle rotation
theta = np.random.uniform(-np.pi, np.pi)
k1 = np.random.uniform(0, 1)
k2 = np.random.uniform(0, 1)
k3 = np.random.uniform(0, 1)
k = np.array([k1, k2, k2]).reshape(3, 1)

axangRot = rbt.conv_axang_to_rotmat(theta, k)
thet, kk = rbt.conv_rotmat_to_axang(axangRot)
print("==>> theta original: \n", theta)
print("==>> k original: \n", k)
print("==>> theta inverse problem: \n", thet)
print("==>> k inverse problem: \n", kk)

# quaternion
q0 = np.random.uniform(0, 1)
q1 = np.random.uniform(0, 1)
q2 = np.random.uniform(0, 1)
q3 = np.random.uniform(0, 1)
quatt = np.array([q0, q1, q2, q3]).reshape(4, 1)
quatt = quatt / np.linalg.norm(quatt)

rotfromqt = rbt.conv_quat_to_rotmat(quatt)
qttt = rbt.conv_rotmat_to_quat(rotfromqt)
print("==>> quaternion original: \n", quatt)
print("==>> quaternion inverse problem: \n", qttt)

q = rbt.conv_axang_to_quat(theta, k)
theta_ori, n_ori = rbt.conv_quat_to_axang(q)
print("==>> theta_ori: \n", theta_ori)
print("==>> n_ori: \n", n_ori)

# skew matrix
x = np.array([[1], [0], [0]])
print(rbt.vec_to_skew(x))

# tool0 to camera_link
roty_n90 = rbt.roty(-np.pi / 2)
rotz_p90 = rbt.rotz(np.pi / 2)
rot = rotz_p90 @ roty_n90
r = R.from_matrix(rot)
rq = r.as_quat()
print(f"==>> rq: \n{rq}")  # xyzw
