import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

np.random.seed(9)


def generate_bezier_trajectory(theta_s, theta_g, N):
    # Bezier curve
    t = np.linspace(0, 1, N)
    P0 = theta_s.reshape(2, 1)
    P1 = np.array([-3, 3]).reshape(2, 1)
    P2 = theta_g.reshape(2, 1)

    trajectory = ((1 - t) ** 2) * P0 + (2 * (1 - t) * t) * P1 + (t**2) * P2
    if print_log:
        ic(trajectory.shape)
        ic(trajectory)

    return trajectory.T


def generate_lerp_trajectory(theta_s, theta_g, N):
    trajectory = np.linspace(theta_s, theta_g, num=N, endpoint=True)

    if print_log:
        ic(trajectory.shape)
        ic(trajectory)

    return trajectory


def add_noise_to_trajectory(trajectory):
    noise = 0.05 * np.random.uniform(-np.pi, np.pi, size=trajectory.shape)
    trajectory = trajectory + noise

    if print_log:
        ic(trajectory.shape)
        ic(trajectory)

    return trajectory


def compute_finitediff_A_matrix(N):
    # diff_rule = [0, 0, -1, 1, 0, 0, 0]
    # diff_rule = [0, 0, 1, -2, 1, 0, 0]  # start with -2
    diff_rule = [0, 1, -2, 1, 0, 0, 0]  # start with the last 1
    # diff_rule = [0, 0, 0, 1, -2, 1, 0]  # start with the first 1
    half_length = len(diff_rule) // 2
    A = np.zeros([N, N])
    for i in range(0, N):
        for j in range(-half_length, half_length):
            index = i + j
            if index >= 0 and index < N:
                A[i, index] = diff_rule[j + half_length]

    if print_log:
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

    if print_log:
        ic(A.shape)
        ic(A)

    return A


def compute_R_and_inverse(A, R_scaler):
    R = A.T @ A
    R_inv = np.linalg.inv(R)

    # scale covariance matrix, from stomp cpp code
    m = np.abs(R_inv).max()
    R_inv_scaled = R_scaler * (R_inv / m)

    if print_log:
        ic(R.shape)
        ic(R)
        ic(R_inv.shape)
        ic(R_inv)
        ic(R_inv_scaled.shape)
        ic(R_inv_scaled)

    return R, R_inv, R_inv_scaled


def compute_M_matrix(N, R_inv):
    # M = R_inv with each column scaled such that the maximum element is 1/N
    max_element = R_inv.max(axis=0)
    scaletoval = 1.0 / N
    scale = scaletoval / max_element
    M = scale * R_inv

    # zeroing out first and last rows
    M[0] = 0.0
    M[0, 0] = 1.0
    M[-1] = 0.0
    M[-1, -1] = 1.0

    if print_log:
        ic(M.shape)
        ic(M)

    return M


def generate_noisy_trajectories(K, stddev, R_inv, trajectory):
    (N, d) = trajectory.shape

    # generate raw noise
    epsilon = np.random.multivariate_normal(mean=np.zeros(N), cov=R_inv, size=(K, d))  # K, d, N
    epsilon = stddev * epsilon.transpose(0, 2, 1)  # K, N, d # scale with std per each joint

    # zeroing out the start and end noise values
    epsilon[:, 0, :] = 0.0
    epsilon[:, -1, :] = 0.0

    # create K noisy trajectories add each generated noise to trajectory
    noisy_trajectories = epsilon + trajectory

    if print_log:
        ic(epsilon.shape)
        ic(epsilon)
        ic(noisy_trajectories.shape)
        ic(noisy_trajectories)

    return noisy_trajectories, epsilon


def compute_traj_obscost(noisy_trajectories):
    bodytoobsfulltraj = noisy_trajectories - obscxy.flatten()
    dminfull = np.linalg.norm(bodytoobsfulltraj, axis=2) - obsr - bodyr
    costfull = np.maximum(clearance + bodyr - dminfull, 0)
    costfully = np.repeat(costfull[:, :, np.newaxis], 2, axis=2)
    return costfully


def obstacle_cost(noisy_trajectories):  # obstacle cost is joint together
    # obscost = np.zeros(noisy_trajectories.shape)
    obscost = compute_traj_obscost(noisy_trajectories)
    return obscost


def constraint_cost(noisy_trajectories):  # constraint cost is joint together
    conscost = np.zeros((noisy_trajectories.shape))
    return conscost


def torque_cost(noisy_trajectories):  # torque cost is individual joint
    # torqcost = np.zeros(noisy_trajectories.shape))
    # change to joint displacement instead. because I dont have a dynamic model.
    cost = np.diff(noisy_trajectories, axis=1, prepend=0.0)
    cost[:, 0, :] = 0.0
    torqcost = np.abs(cost)

    # torqcost = np.abs(cost).sum(axis=2).T

    if print_log:
        ic(cost.shape)
        ic(cost)
        ic(torqcost.shape)
        ic(torqcost)

    return torqcost


def compute_cost_S(noisy_trajectories):
    # a point in trajectory must have aleast some concept of cost.
    # that cost will be use to compute probability and later determine the update value.
    qo = obstacle_cost(noisy_trajectories)
    qc = constraint_cost(noisy_trajectories)
    qt = torque_cost(noisy_trajectories)
    S = qo + qc + qt

    if print_log:
        ic(S.shape)
        ic(S)

    return S


def compute_probability_P(S, h):
    # In Eq 11
    Smin = S.min(axis=0, keepdims=True)  # K trajectory wise
    Smax = S.max(axis=0, keepdims=True)  # K trajectory wise

    numerator = np.exp(-h * (S - Smin) / (Smax - Smin))  # inverse softmax
    denominator = np.sum(numerator, axis=0, keepdims=True)  # sum over k trajectories
    P = numerator / denominator
    P[:, 0, :] = 0.0  # avoid first element 0 to NaN
    P[:, -1, :] = 0.0  # zero out last element

    if print_log:
        ic(Smin.shape)
        ic(Smin)
        ic(Smax.shape)
        ic(Smax)
        ic(P.shape)
        ic(P)

    return P


def compute_noisy_update(P, epsilon):
    # from a probability-weighted (convex combination) of noisy parameter from that time step
    # Preshape = P[np.newaxis, :, :]
    # Preshape = Preshape.transpose(2, 1, 0)
    # P_epsilon = Preshape * epsilon
    # delta_trajectory_noise = P_epsilon.sum(axis=0)

    P_epsilon = P * epsilon
    delta_trajectory_noise = P_epsilon.sum(axis=0)
    # delta_trajectory_noise[-1] = 0.0  # pin the last node exactly at the goal state, we don't want to update it

    if print_log:
        ic(P_epsilon.shape)
        ic(P_epsilon)
        ic(delta_trajectory_noise.shape)
        ic(delta_trajectory_noise)

    return delta_trajectory_noise


def smooth_noisy(M, delta_trajectory_noise):
    # compute update value scaling ensures that no updated parameter exceeds the range that was explored in the noisy trajectories.
    # ensures that the updated trajectory remains smooth.
    delta_trajectory_smoothed = M @ delta_trajectory_noise
    # delta_trajectory_smoothed = delta_trajectory_noise

    if print_log:
        ic(delta_trajectory_smoothed.shape)
        ic(delta_trajectory_smoothed)

    return delta_trajectory_smoothed


def update_trajectory(trajectory, delta_trajectory_smoothed):
    trajectory_new = trajectory + delta_trajectory_smoothed  # column-wise addition

    if print_log:
        ic(trajectory_new.shape)
        ic(trajectory_new)

    return trajectory_new


def unupdate_trajectory(trajectory, delta_trajectory_smoothed):
    trajectory_new = trajectory - delta_trajectory_smoothed  # column-wise subtraction

    if print_log:
        ic(trajectory_new.shape)
        ic(trajectory_new)

    return trajectory_new


def compute_trajectory_distance_cost(trajectory):
    diffs = np.diff(trajectory, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return np.sum(distances)


def compute_trajectory_obstacle_cost(trajectory):
    bodytoobs = trajectory - obscxy.flatten()
    dmin = np.linalg.norm(bodytoobs, axis=1) - obsr - bodyr
    cost = np.maximum(clearance + bodyr - dmin, 0)
    return np.sum(cost)


def compute_trajectory_cost_Q(trajectory, R):
    smoothness_cost = 0.5 * np.trace(trajectory.T @ R @ trajectory)
    trajectory_cost = compute_trajectory_distance_cost(trajectory) + compute_trajectory_obstacle_cost(trajectory)
    totalcost = trajectory_cost + smoothness_cost

    if print_log:
        ic(smoothness_cost)
        ic(trajectory_cost)
        ic(totalcost)

    return totalcost


def optimize(theta_s, theta_g, N, R_scaler, K, stddev, h, num_iterations):
    A = compute_finitediff_A_matrix_new(N)  # we must have index [0,0] and [-1,-1] equal to 1. otherwise, it is gonna be look weird
    # A = compute_finitediff_A_matrix(N)

    # trajectory = generate_bezier_trajectory(theta_s, theta_g, N)
    # trajectory_opt = generate_bezier_trajectory(theta_s, theta_g, N)

    trajectory = generate_lerp_trajectory(theta_s, theta_g, N)
    trajectory_opt = generate_lerp_trajectory(theta_s, theta_g, N)

    R, R_inv, R_inv_scaled = compute_R_and_inverse(A, R_scaler)
    M = compute_M_matrix(N, R_inv)
    cost = compute_trajectory_cost_Q(trajectory_opt, R)

    cost_history = []
    for ii in range(num_iterations):

        noisy_trajectories, epsilon = generate_noisy_trajectories(K, stddev, R_inv_scaled, trajectory_opt)
        S = compute_cost_S(noisy_trajectories)
        P = compute_probability_P(S, h)
        delta_trajectory_noise = compute_noisy_update(P, epsilon)
        delta_trajectory_smoothed = smooth_noisy(M, delta_trajectory_noise)
        trajectory_opt = update_trajectory(trajectory_opt, delta_trajectory_smoothed)

        current_cost = compute_trajectory_cost_Q(trajectory_opt, R)

        if current_cost < cost:  # check whether to keep the updated trajectory or revert back
            cost = current_cost
        else:  # if it become worst, go back to the previous trajectory
            trajectory_opt = unupdate_trajectory(trajectory_opt, delta_trajectory_smoothed)

        cost_history.append(cost)

    return trajectory, trajectory_opt, R_inv_scaled, noisy_trajectories, cost_history


if __name__ == "__main__":
    # parameters
    N = 60  # number of waypoints 10
    K = 30  # number of noisy trajectory to be generated 6
    d = 2  # system dimension
    h = 10  # sensitivity scale
    tolerance = 0.1  # terminate upon the value is below
    # R_scaler = 100  # scale R inverse
    # stddev = [0.05] * d  # 2 joints
    R_scaler = 10  # scale R inverse
    stddev = [0.5] * d  # 2 joints
    # stddev: this is the degree of noise that can be applied to the joints. ex: stddev = [0.05, 0.8, 0.8, 0.4, 0.4, 0.4] # 6 joints
    # Each value in this array is the amplitude of the noise applied to the joint at that position in the array.
    # For instace, the leftmost value in the array will be the value used to set the noise of the first joint of the robot (panda_joint1 in our case).
    # The dimensionality of this array should be equal to the number of joints in the planning group name.
    # Larger “stddev” values correspond to larger motions of the joints.
    num_iterations = 1000
    times = np.linspace(0, N, num=N)
    print_log = False

    # environment
    theta_s = np.array([-2.0, -2.0])
    theta_g = np.array([2.0, 2.0])
    xlim = [-np.pi, np.pi]
    ylim = [-np.pi, np.pi]
    # obstacle
    obscxy = np.array([0.0, 0.0]).reshape(2, 1)
    obsr = 0.7
    bodyr = 0.2
    clearance = 0.1

    # stomp
    if False:
        A = compute_finitediff_A_matrix_new(N)  # we must have index [0,0] and [-1,-1] equal to 1. otherwise, it is gonna be look weird
        # A = compute_finitediff_A_matrix(N)
        # trajectory_init = generate_lerp_trajectory(theta_s, theta_g, N)
        # trajectory_init = generate_bezier_trajectory(theta_s, theta_g, N)
        # R, R_inv, R_inv_scaled = compute_R_and_inverse(A, R_scaler)
        # M = compute_M_matrix(N, R_inv)

        # noisy_trajectories, epsilon = generate_noisy_trajectories(K, stddev, R_inv_scaled, trajectory_init)
        # S = compute_cost_S(noisy_trajectories)
        # P = compute_probability_P(S, h)
        # delta_trajectory_noise = compute_noisy_update(P, epsilon)
        # delta_trajectory_smoothed = smooth_noisy(M, delta_trajectory_noise)
        # trajectory_new = update_trajectory(trajectory_init, delta_trajectory_smoothed)
        # cost = compute_trajectory_cost_Q(trajectory_new, R)

        # t = np.array([0, 2, 10, 22, 35]).reshape(-1, 1)
        # ic(t)
        # ic(A @ t)
        # R = A.T @ A
        # ic(R)
        # ic(t.T @ R @ t)

    else:
        trajectory_init, trajectory_new, R_inv_scaled, noisy_trajectories, cost_history = optimize(theta_s, theta_g, N, R_scaler, K, stddev, h, num_iterations)

    if True:
        # plot trajectory
        if True:
            fig1, ax1 = plt.subplots()
            ax1.plot(trajectory_init[:, 0], trajectory_init[:, 1], "k--", linewidth=4, label="Original Trajectory")
            ax1.plot(theta_s[0], theta_s[1], "ro", label="Start")
            ax1.plot(theta_g[0], theta_g[1], "bo", label="Goal")
            ax1.set_xlim(xlim)
            ax1.set_ylim(ylim)
            ax1.set_title("Trajectory")
            ax1.set_xlabel("Theta 1")
            ax1.set_ylabel("Theta 2")
            ax1.set_aspect("equal", adjustable="box")
            ax1.legend()

            circle = plt.Circle(obscxy, obsr, color="r", alpha=0.5)
            ax1.add_patch(circle)

            if True:
                fig11, axs11 = plt.subplots(d, 1, sharex=True)
                for i in range(d):
                    axs11[i].plot(times, trajectory_init[..., i], "k--", label=f"Original {i+1}")
                    axs11[i].plot(times, [-np.pi] * len(times), "m", label=f"Joint Min : {-np.pi:.3f}")
                    axs11[i].plot(times, [np.pi] * len(times), "c", label=f"Joint Max : {np.pi:.3f}")
                    axs11[i].set_ylabel(f"theta {i+1}")
                    axs11[i].set_xlim(times[0], times[-1])
                    axs11[i].legend(loc="upper right")
                    axs11[i].grid(True)
                axs11[-1].set_xlabel("Time")

        # plot R_inv
        if True:
            fig2, ax2 = plt.subplots()
            for i in range(N):
                ax2.plot(R_inv_scaled[i, :], label=f"Row {i+1}", alpha=0.5)
            ax2.set_title("Inverse Covariance Matrix Rows (R^-1)")
            ax2.set_xlabel("Timestep Index")
            ax2.set_ylabel("Value")
            ax2.legend()

        # plot noisy trajectory
        if True:
            fig3, ax3 = plt.subplots()
            ax3.plot(theta_s[0], theta_s[1], "ro", label="Start")
            ax3.plot(theta_g[0], theta_g[1], "bo", label="Goal")
            for i in range(K):
                ax3.plot(noisy_trajectories[i, :, 0], noisy_trajectories[i, :, 1], alpha=0.5)
            ax3.set_xlim(xlim)
            ax3.set_ylim(ylim)
            ax3.set_title("Noise Trajectories")
            ax3.set_xlabel("Theta 1")
            ax3.set_ylabel("Theta 2")
            ax3.set_aspect("equal", adjustable="box")
            ax3.legend()

            circle = plt.Circle(obscxy, obsr, color="r", alpha=0.5)
            ax3.add_patch(circle)

            if True:
                fig33, axs33 = plt.subplots(d, 1, sharex=True)
                for i in range(d):
                    for j in range(noisy_trajectories.shape[0]):
                        axs33[i].plot(times, noisy_trajectories[j, ..., i], alpha=0.5)
                    axs33[i].plot(times, [-np.pi] * len(times), "m", label=f"Joint Min : {-np.pi:.3f}")
                    axs33[i].plot(times, [np.pi] * len(times), "c", label=f"Joint Max : {np.pi:.3f}")
                    axs33[i].set_ylabel(f"theta {i+1}")
                    axs33[i].set_xlim(times[0], times[-1])
                    axs33[i].legend(loc="upper right")
                    axs33[i].grid(True)
                axs33[-1].set_xlabel("Time")

        # plot cost history
        if True:
            fig4, ax4 = plt.subplots()
            ax4.plot(cost_history)
            ax4.set_title("Cost History")
            ax4.set_xlabel("Iteration")
            ax4.set_ylabel("Cost")

        # plot comparison
        if True:
            fig5, ax5 = plt.subplots()
            ax5.plot(theta_s[0], theta_s[1], "ro", label="Start")
            ax5.plot(theta_g[0], theta_g[1], "bo", label="Goal")
            ax5.plot(trajectory_init[:, 0], trajectory_init[:, 1], "k--", linewidth=4, label="Original Trajectory")
            ax5.plot(trajectory_new[:, 0], trajectory_new[:, 1], "r+", linewidth=4, label="Optimal Trajectory")

            for point in trajectory_new:
                circle = plt.Circle((point[0], point[1]), radius=bodyr + clearance, color="b", fill=False, alpha=0.5)
                ax5.add_patch(circle)
                circle1 = plt.Circle((point[0], point[1]), radius=bodyr, color="r", fill=False, alpha=0.5)
                ax5.add_patch(circle1)

            ax5.set_xlim(xlim)
            ax5.set_ylim(ylim)
            ax5.set_title("Trajectory Comparison")
            ax5.set_xlabel("Theta 1")
            ax5.set_ylabel("Theta 2")
            ax5.set_aspect("equal", adjustable="box")
            ax5.legend()

            circle = plt.Circle(obscxy, obsr, color="r", alpha=0.5)
            ax5.add_patch(circle)

            if True:
                fig55, axs55 = plt.subplots(d, 1, sharex=True)
                for i in range(d):
                    axs55[i].plot(times, trajectory_init[..., i], "k--", label=f"Original {i+1}")
                    axs55[i].plot(times, trajectory_new[..., i], "r+", label=f"Optimal {i+1}")
                    axs55[i].plot(times, [-np.pi] * len(times), "m", label=f"Joint Min : {-np.pi:.3f}")
                    axs55[i].plot(times, [np.pi] * len(times), "c", label=f"Joint Max : {np.pi:.3f}")
                    axs55[i].set_ylabel(f"theta {i+1}")
                    axs55[i].set_xlim(times[0], times[-1])
                    axs55[i].legend(loc="upper right")
                    axs55[i].grid(True)
                axs55[-1].set_xlabel("Time")

        plt.show()
