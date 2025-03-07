import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from stomp_linkpoint import compute_traj_obscost_per_N, plot_arm

# import logging
# logging.basicConfig(filename='example.log', level=logging.DEBUG)
np.set_printoptions(linewidth=2000, suppress=True)
np.random.seed(9)


# parameters
# N = 10  # number of waypoints
# K = 6  # number of noisy trajectory to be generated
N = 100  # number of waypoints
K = 30  # number of noisy trajectory to be generated
d = 2  # system dimension
h = 10  # sensitivity scale
tolerance = 0.1  # terminate upon the value is below
R_scaler = 0.001  # scale R inverse
stddev = [0.05, 0.05]  # 2 joints
# stddev: this is the degree of noise that can be applied to the joints. ex: stddev = [0.05, 0.8, 0.8, 0.4, 0.4, 0.4] # 6 joints
# Each value in this array is the amplitude of the noise applied to the joint at that position in the array.
# For instace, the leftmost value in the array will be the value used to set the noise of the first joint of the robot (panda_joint1 in our case).
# The dimensionality of this array should be equal to the number of joints in the planning group name.
# Larger “stddev” values correspond to larger motions of the joints.
num_iterations = 4000


def generate_initial_trajectory():
    # theta_s = np.array([0.0] * d)
    # theta_g = np.array([np.pi] * d)
    theta_s = np.array([0.0, 0.0])
    theta_g = np.array([np.pi, 0.0])
    trajectory = np.linspace(theta_s, theta_g, num=N, endpoint=True)
    noise = 0.05 * np.random.uniform(-np.pi, np.pi, size=(N, d))
    trajectory = trajectory + noise
    ic(trajectory.shape)
    ic(trajectory)

    return trajectory


def compute_finitediff_A_matrix(N):
    # diff_rule = [0, 0, -1, 1, 0, 0, 0]
    diff_rule = [0, 0, 1, -2, 1, 0, 0]  # start with -2
    # diff_rule = [0, 1, -2, 1, 0, 0, 0]  # start with the last 1
    # diff_rule = [0, 0, 0, 1, -2, 1, 0]  # start with the first 1
    half_length = len(diff_rule) // 2
    A = np.zeros([N, N])
    for i in range(0, N):
        for j in range(-half_length, half_length):
            index = i + j
            if index >= 0 and index < N:
                A[i, index] = diff_rule[j + half_length]
    ic(A.shape)
    ic(A)
    return A


def compute_finitediff_A_matrix_new(N):
    diff_rule = [0, 1, -2, 1, 0, 0, 0]  # start with the last 1
    half_length = len(diff_rule) // 2
    A = np.zeros([N + 2, N])
    for i in range(0, N + 2):
        for j in range(-half_length, half_length):
            index = i + j
            if index >= 0 and index < N:
                A[i, index] = diff_rule[j + half_length]
    ic(A.shape)
    ic(A)
    return A


def compute_R_and_inverse(A):
    R = A.T @ A
    R_inv = np.linalg.inv(R)

    # scale covariance matrix, from stomp cpp code
    m = np.abs(R_inv).max()
    R_inv_scale = R_scaler * (R_inv / m)

    ic(R.shape)
    ic(R)
    ic(R_inv.shape)
    ic(R_inv)
    ic(R_inv_scale.shape)
    ic(R_inv_scale)

    return R, R_inv, R_inv_scale


def compute_M_matrix(N, R_inv):
    # M = R_inv with each column scaled such that the maximum element is 1/N
    m = R_inv.max(axis=0)
    scaletoval = 1.0 / N
    scale = scaletoval / m
    M = scale * R_inv

    # zeroing out first and last rows
    M[0] = 0.0
    M[0, 0] = 1.0
    M[-1] = 0.0
    M[-1, -1] = 1.0

    ic(M.shape)
    ic(M)

    return M


def generate_noisy_trajectories(N, K, d, R_inv_scale, trajectory):
    # generate raw noise
    epsilon = np.random.multivariate_normal(mean=np.zeros(N), cov=R_inv_scale, size=(K, d))  # K, d, N
    epsilon = stddev * epsilon.transpose(0, 2, 1)  # K, N, d # scale with std per each joint

    # zeroing out the start and end noise values
    epsilon[:, 0, :] = 0.0
    epsilon[:, -1, :] = 0.0

    # create K noisy trajectories add each generated noise to trajectory
    noisy_trajectories = epsilon + trajectory

    ic(epsilon.shape)
    ic(epsilon)
    ic(noisy_trajectories.shape)
    ic(noisy_trajectories)

    return noisy_trajectories, epsilon


def obstacle_cost(noisy_trajectories):
    obscost = np.zeros((N, K))
    # for i in range(K):
    #     q = compute_traj_obscost_per_N(noisy_trajectories[i])
    #     obscost[:, i] = q
    return obscost


def constraint_cost(noisy_trajectories):
    conscost = np.zeros((N, K))
    return conscost


def torque_cost(noisy_trajectories):
    # torqcost = np.zeros((N, K))
    # change to joint displacement instead. because i dont have a dynamic model.
    e = np.diff(noisy_trajectories, axis=1, prepend=0.0)
    e[:, 0, :] = 0.0
    torqcost = np.abs(e).sum(axis=2).T
    return torqcost


def compute_cost_S_as_matrix(noisy_trajectories):
    qo = obstacle_cost(noisy_trajectories)
    qc = constraint_cost(noisy_trajectories)
    qt = torque_cost(noisy_trajectories)
    S = qo + qc + qt  # N X K

    ic(S.shape)
    ic(S)

    return S


def compute_probability_P_as_matrix(S, h, N):
    # In algoritm table
    # lamdas = 1 / h
    # Se = np.exp(-lamdas * S)
    # SKsum = Se.sum(axis=1).reshape(N, -1)
    # P = Se / SKsum

    # In Eq 11
    Smin = S.min(axis=1, keepdims=True)  # K trajectory wise
    Smax = S.max(axis=1, keepdims=True)
    numerator = np.exp(-h * (S - Smin) / (Smax - Smin))  # inverse softmax
    denominator = np.sum(numerator, axis=1, keepdims=True)
    P = numerator / denominator
    P[0, :] = 0.0  # avoid first element 0 to NaN

    ic(P.shape)
    ic(P)

    return P


def compute_noisy_update(P, epsilon):
    # from a probability-weighted (convex combination) of noisy parameter from that time step
    Preshape = P[np.newaxis, :, :]
    Preshape = Preshape.transpose(2, 1, 0)
    P_epsilon = Preshape * epsilon
    delta_trajectory_noise = P_epsilon.sum(axis=0)

    # Preshape = P[:, :, np.newaxis]
    # epsilonreshape = epsilon.transpose(1, 0, 2)
    # P_epsilon = Preshape * epsilonreshape
    # delta_trajectory_noise = P_epsilon.sum(axis=1)
    # delta_trajectory_noise[-1] = 0.0  # pin the last node exactly at the goal state, we don't want to update it

    ic(Preshape.shape)
    ic(Preshape)
    # ic(epsilonreshape.shape)
    # ic(epsilonreshape)
    ic(P_epsilon.shape)
    ic(P_epsilon)
    ic(delta_trajectory_noise.shape)
    ic(delta_trajectory_noise)

    return delta_trajectory_noise


def trajectory_update_term(M, delta_trajectory_noise):
    # compute update value scaling ensures that no updated parameter exceeds the range that was explored in the noisy trajectories.
    # ensures that the updated trajectory remains smooth.
    delta_trajectory = M @ delta_trajectory_noise
    ic(delta_trajectory.shape)
    ic(delta_trajectory)
    return delta_trajectory


def compute_trajectory_cost_Q(trajectory, R):
    smoothness_loss = 0.5 * np.trace(trajectory.T @ R @ trajectory)
    sdcqtotal = compute_traj_obscost_per_N(trajectory).sum()
    totalcost = sdcqtotal + smoothness_loss
    ic(totalcost)
    return totalcost


def optimize():
    trajectory = generate_initial_trajectory()

    trajectory_new = generate_initial_trajectory()
    A = compute_finitediff_A_matrix_new(N)
    R, R_inv, R_inv_scale = compute_R_and_inverse(A)
    # M = compute_M_matrix(N, R_inv)
    M = compute_M_matrix(N, R_inv_scale)

    cost = compute_trajectory_cost_Q(trajectory_new, R)

    # while compute_trajectory_cost_Q(trajectory) > tolerance:
    for ii in range(num_iterations):
        noisy_trajectories, epsilon = generate_noisy_trajectories(N, K, d, R_inv_scale, trajectory_new)

        # compute cost and probability contribution
        S = compute_cost_S_as_matrix(noisy_trajectories)
        P = compute_probability_P_as_matrix(S, h, N)

        # compute update term
        delta_trajectory_noise = compute_noisy_update(P, epsilon)
        delta_trajectory = trajectory_update_term(M, delta_trajectory_noise)

        # update trajectory
        trajectory_new = trajectory_new + delta_trajectory
        current_cost = compute_trajectory_cost_Q(trajectory_new, R)

        # check whether to keep the updated trajectory or revert back
        if current_cost < cost:
            cost = current_cost
        else:  # if it become worst, go back to the previous trajectory
            trajectory_new = trajectory_new - delta_trajectory
        ic(current_cost)
    return trajectory, trajectory_new, R_inv_scale, noisy_trajectories, epsilon


# # single test -----------------------------------------------------------------------------------------
# # we must have index 0,0 and -1,-1 to be 1. otherwise, it is gonna be look weird
# # A = compute_finitediff_A_matrix(N)
# A = compute_finitediff_A_matrix_new(N)
# trajectory = generate_initial_trajectory()
# R, R_inv, R_inv_scale = compute_R_and_inverse(A)
# noisy_trajectories, epsilon = generate_noisy_trajectories(N, K, d, R_inv_scale, trajectory)
# S = compute_cost_S_as_matrix(noisy_trajectories)
# P = compute_probability_P_as_matrix(S, h, N)
# delta_trajectory_noise = compute_noisy_update(P, epsilon)
# M = compute_M_matrix(N, R_inv)
# delta_trajectory = trajectory_update_term(M, delta_trajectory_noise)
# trajectory_new = trajectory + delta_trajectory  # column-wise addition
# cost = compute_trajectory_cost_Q(trajectory_new, R)


# loop test -----------------------------------------------------------------------------------------
ic.disable()
trajectory, trajectory_new, R_inv_scale, noisy_trajectories, epsilon = optimize()
ic.enable()
ic(trajectory_new)


# raise SystemExit(0)


# Plot ------------------------------------------------------------------------------------------------
def plot_noise():
    times = np.linspace(0, N, num=N)
    fig, axs = plt.subplots(1, 1, figsize=(10, 15))
    for i in range(epsilon.shape[0]):
        for j in range(d):
            axs.plot(times, epsilon[i, :, j])
    axs.plot(times, [-np.pi] * len(times), "m", label=f"Joint Min : {-np.pi:.3f}")
    axs.plot(times, [np.pi] * len(times), "c", label=f"Joint Max : {np.pi:.3f}")
    axs.set_xlim(times[0], times[-1])
    axs.legend(loc="upper right")
    axs.grid(True)
    axs.set_ylabel(f"noise")
    axs.set_xlabel("Time")
    fig.suptitle("All noise epsilon")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_trajectory():
    times = np.linspace(0, N, num=N)
    fig, axs = plt.subplots(d, 1, figsize=(10, 15), sharex=True)
    for i in range(d):
        axs[i].plot(times, trajectory[..., i], "g-", label=f"Joint Position {i+1}")
        axs[i].plot(times, [-np.pi] * len(times), "m", label=f"Joint Min : {-np.pi:.3f}")
        axs[i].plot(times, [np.pi] * len(times), "c", label=f"Joint Max : {np.pi:.3f}")
        axs[i].set_ylabel(f"theta {i+1}")
        axs[i].set_xlim(times[0], times[-1])
        axs[i].legend(loc="upper right")
        axs[i].grid(True)
    axs[-1].set_xlabel("Time")
    fig.suptitle("Original trajectory")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_noisy_trajectory():
    times = np.linspace(0, N, num=N)
    fig, axs = plt.subplots(d, 1, figsize=(10, 15), sharex=True)
    for i in range(d):
        axs[i].plot(times, trajectory[..., i], "g-", label=f"OG Joint Position {i+1}")
        for j in range(noisy_trajectories.shape[0]):
            axs[i].plot(times, noisy_trajectories[j, ..., i], "r-")
        axs[i].plot(times, [-np.pi] * len(times), "m", label=f"Joint Min : {-np.pi:.3f}")
        axs[i].plot(times, [np.pi] * len(times), "c", label=f"Joint Max : {np.pi:.3f}")
        axs[i].set_ylabel(f"theta {i+1}")
        axs[i].set_xlim(times[0], times[-1])
        axs[i].legend(loc="upper right")
        axs[i].grid(True)
    axs[-1].set_xlabel("Time")
    fig.suptitle("Noisy Trajectory")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_trajectory_after():
    times = np.linspace(0, N, num=N)
    fig, axs = plt.subplots(d, 1, figsize=(10, 15), sharex=True)
    for i in range(d):
        axs[i].plot(times, trajectory[..., i], "g-", label=f"Old Joint Position {i+1}")
        axs[i].plot(times, trajectory_new[..., i], "r-", label=f"Joint Position {i+1}")
        axs[i].plot(times, [-np.pi] * len(times), "m", label=f"Joint Min : {-np.pi:.3f}")
        axs[i].plot(times, [np.pi] * len(times), "c", label=f"Joint Max : {np.pi:.3f}")
        axs[i].set_ylabel(f"theta {i+1}")
        axs[i].set_xlim(times[0], times[-1])
        axs[i].legend(loc="upper right")
        axs[i].grid(True)
    axs[-1].set_xlabel("Time")
    fig.suptitle("Update New Trajectory")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_R_inverse():
    plt.figure(figsize=(10, 6))
    for i in range(N):
        plt.plot(R_inv_scale[i, :], label=f"Row {i+1}", alpha=0.5)
    plt.title("Inverse Covariance Matrix Rows (R^-1)")
    plt.xlabel("Timestep Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()


plot_noise()
plot_trajectory()
plot_noisy_trajectory()
plot_trajectory_after()
plot_R_inverse()


import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))
from matplotlib import animation


class PlanarRobot:

    def __init__(self):
        self.alpha1 = 0
        self.alpha2 = 0
        self.d1 = 0
        self.d2 = 0
        self.a1 = 1
        self.a2 = 1

    def forward_kinematic(self, theta, return_link_pos=False):
        theta1 = theta[0, 0]
        theta2 = theta[1, 0]

        x = self.a1 * np.cos(theta1) + self.a2 * np.cos(theta1 + theta2)
        y = self.a1 * np.sin(theta1) + self.a2 * np.sin(theta1 + theta2)

        if return_link_pos:

            # option for return link end pose. normally used for collision checking
            link_end_pose = []
            link_end_pose.append([0, 0])

            # link 1 pose
            x1 = self.a1 * np.cos(theta1)
            y1 = self.a1 * np.sin(theta1)
            link_end_pose.append([x1, y1])

            # link 2 pose
            x2 = self.a1 * np.cos(theta1) + self.a2 * np.cos(theta1 + theta2)
            y2 = self.a1 * np.sin(theta1) + self.a2 * np.sin(theta1 + theta2)
            link_end_pose.append([x2, y2])

            return link_end_pose

        else:
            return np.array([[x], [y]])


robot = PlanarRobot()

# obstacle
cobs = np.array([0.0, 1.5]).reshape(2, 1)  # x y
robs = 0.2  # r


def play_back_path(path, animation):  # path format (2,n)
    # plot task space
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_aspect("equal")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    circle1 = plt.Circle(cobs, robs, color="r")
    plt.gca().add_patch(circle1)

    # plot animation link
    (robotLinks,) = ax.plot([], [], color="indigo", linewidth=5, marker="o", markerfacecolor="r")

    def update(frame):
        link = robot.forward_kinematic(path[:, frame].reshape(2, 1), return_link_pos=True)
        robotLinks.set_data([link[0][0], link[1][0], link[2][0]], [link[0][1], link[1][1], link[2][1]])

    animation = animation.FuncAnimation(fig, update, frames=(path.shape[1]), interval=100, repeat=False)
    plt.show()


play_back_path(trajectory_new.T, animation)
