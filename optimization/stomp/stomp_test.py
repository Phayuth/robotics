group_name = "panda_arm"
optimization = None
num_timesteps = 60
num_iterations = 40
num_iterations_after_valid = 0
num_rollouts = 30
max_rollouts = 30
initialization_method = 1  # [1 = LINEAR_INTERPOLATION, 2 = CUBIC_POLYNOMIAL, 3 = MININUM_CONTROL_COST]
control_cost_weight = 0.0
task = None


class noise_generator:
    name = "stomp_moveit/NormalDistributionSampling"
    stddev = [0.05, 0.8, 1.0, 0.8, 0.4, 0.4, 0.4]


class cost_functions:
    name = "stomp_moveit/CollisionCheck"
    collision_penalty = 1.0
    cost_weight = 1.0
    kernel_window_percentage = 0.2
    longest_valid_joint_move = 0.05


class noisy_filters:
    name = "stomp_moveit/JointLimits"
    lock_start = True
    lock_goal = True
    name = "stomp_moveit/MultiTrajectoryVisualization"
    line_width = 0.02
    rgb = [255, 255, 0]
    marker_array_topic = "stomp_trajectories"
    marker_namespace = "noisy"


class update_filters:
    name = "stomp_moveit/PolynomialSmoother"
    poly_order = 6
    name = "stomp_moveit/TrajectoryVisualization"
    line_width = 0.05
    rgb = [0, 191, 255]
    error_rgb = [255, 0, 0]
    publish_intermediate = True
    marker_topic = "stomp_trajectory"
    marker_namespace = "optimized"


# https://docs.ros.org/en/kinetic/api/moveit_tutorials/html/doc/stomp_planner/stomp_planner_tutorial.html


import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

np.random.seed(9)


def single_normal_rand():
    mu = 0
    sigma = 1
    xrand = np.random.normal(mu, sigma)
    return xrand


def multivariate_normal_rand():
    mus = np.array([2, 0])
    sigx = 1
    sigy = 1
    sigmas = np.array([[sigx**2, 0.5], [0.5, sigy**2]])
    x, y = np.random.multivariate_normal(mus, sigmas, size=(100000)).T
    ic(x.shape)
    ic(y.shape)

    plt.plot(x, y, "b*")
    plt.axis("equal")
    plt.show()


K = 7
N = 10
d = 2


# compute joint displacement ------------------------------
noisy = np.random.randint(low=0, high=10, size=(K, N, d))
ic(noisy.shape)
ic(noisy)

d = np.diff(noisy, axis=1)
ic(d)
ic(d.shape)

e = np.diff(noisy, axis=1, prepend=0)
e[:, 0, :] = 0.0
ic(e)
ic(e.shape)

f = np.abs(e).sum(axis=2).T
# f = e.sum(axis=2).T
ic(f)
ic(f.shape)


# # compute P-----------------------------------------------
# S = np.random.randint(low=0,high=10,size=(N, K))
# ic(S)

# # K trajectory wise
# Smin = S.min(axis=1, keepdims=True)
# Smax = S.max(axis=1, keepdims=True)

# ic(Smin)
# ic(Smax)
# eS = Smax - Smin
# ic(eS)

# ee = (S - Smin) / eS
# ic(ee)

# exp = np.exp(-10*ee)
# ic(exp)


# nn = 15
# epsilon = np.random.randint(low=0, high=10, size=nn)
# ic(epsilon)

# S = np.random.uniform(0, 10, size=(nn))
# ic(S)

# beta = 1.0
# softmax = np.exp(beta * S) / np.sum(np.exp(beta * S))
# ic(softmax)

# h = 1.0
# Smin = S.min()  # K trajectory wise
# Smax = S.max()
# softmax_inv = np.exp(-h * (S - Smin) / (Smax - Smin))
# ic(softmax_inv)


# def stable_softmax(x):
#     z = x - max(x)
#     numerator = np.exp(z)
#     denominator = np.sum(numerator)
#     softmax = numerator/denominator
#     return softmax


# softmaxstable = stable_softmax(S)
# ic(softmaxstable)

ic(noisy.shape)

obscxy = np.array([0.0, 0.0])
obsr = 0.7
br = 0.2
clearence = 0.1

traj1 = noisy[0]
ic(traj1.shape)
ic(traj1)

bodytoobs = traj1 - obscxy
ic(bodytoobs)

dmin = np.linalg.norm(bodytoobs, axis=1) - obsr - br
ic(dmin)

cost = np.maximum(clearence + br - dmin, 0)
ic(cost)


# ic(noisy)

# bodytoobsfulltraj = noisy - obscxy
# ic(bodytoobsfulltraj.shape)
# ic(bodytoobsfulltraj)

# dminfull = np.linalg.norm(bodytoobsfulltraj, axis=2) - obsr - br
# ic(dminfull.shape)
# ic(dminfull)

# costfull = np.maximum(clearence + br - dminfull, 0)
# ic(costfull.shape)
# ic(costfull)

# costfully = np.repeat(costfull[:, :, np.newaxis], 2, axis=2)
# ic(costfully.shape)
# ic(costfully)


def verify_cost_acceleration():
    # sum of squared accelerations along the trajectory
    t1 = np.random.random_integers(0, 9, size=(10, 1))
    ic(t1)
    R = np.eye(10)
    ic(t1.T @ R @ t1)

    t2 = np.random.random_integers(0, 9, size=(10, 1))
    ic(t2)
    ic(t2.T @ R @ t2)

    t = np.hstack((t1, t2))
    ic(t)
    ic(t.T @ R @ t)
