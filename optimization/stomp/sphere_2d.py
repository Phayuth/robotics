import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from icecream import ic
from stomp import STOMP, STOMPUTILS


def compute_traj_obscost(noisy_trajectories):
    # input : noisy_trajectories = K, N, d
    # output : costfully = K, N, d
    bodytoobsfulltraj = noisy_trajectories - obscxy.flatten()
    dminfull = np.linalg.norm(bodytoobsfulltraj, axis=2) - obsr - bodyr
    costfull = np.maximum(clearance + bodyr - dminfull, 0)  # i only have 1 body and 1 obstacle so no sum
    costfully = np.repeat(costfull[:, :, np.newaxis], 2, axis=2)
    return costfully


def compute_trajectory_obstacle_cost(trajectory):
    bodytoobs = trajectory - obscxy.flatten()
    dmin = np.linalg.norm(bodytoobs, axis=1) - obsr - bodyr
    cost = np.maximum(clearance + bodyr - dmin, 0)
    return np.sum(cost)


if __name__ == "__main__":

    K = 50
    N = 100
    d = 2
    h = 10
    tolerance = 0.1
    Rscale = 10
    stddev = [0.095] * 2
    num_iterations = 100

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
    times = np.linspace(0, N, num=N)

    # generate trajectory
    # trajectory = STOMPUTILS.generate_bezier_trajectory(theta_s, theta_g, 100)
    trajectory = STOMPUTILS.generate_lerp_trajectory(theta_s, theta_g, 100)
    stomp = STOMP(
        K=K,
        N=N,
        d=d,
        h=h,
        tolerance=tolerance,
        Rscale=Rscale,
        stddev=stddev,
        num_iterations=num_iterations,
        func_obstacle_cost=compute_traj_obscost,
        func_constraint_cost=STOMPUTILS.constraint_cost,
        func_torque_cost=STOMPUTILS.torque_cost,
        func_obstacle_cost_single=compute_trajectory_obstacle_cost,
        func_constraint_cost_single=STOMPUTILS.compute_trajectory_constraint_cost,
        func_torque_cost_single=STOMPUTILS.compute_trajectory_torque_cost,
        seed=9,
        print_debug=False,
        print_log=False,
    )
    trajectory_opt = stomp.optimize(trajectory)

    # plot comparison
    if True:
        fig5, ax5 = plt.subplots()
        ax5.plot(theta_s[0], theta_s[1], "ro", label="Start")
        ax5.plot(theta_g[0], theta_g[1], "bo", label="Goal")
        ax5.plot(trajectory[:, 0], trajectory[:, 1], "k--", linewidth=4, label="Original Trajectory")
        ax5.plot(trajectory_opt[:, 0], trajectory_opt[:, 1], "r+", linewidth=4, label="Optimal Trajectory")

        for point in trajectory_opt:
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
                axs55[i].plot(times, trajectory[..., i], "k--", label=f"Original {i+1}")
                axs55[i].plot(times, trajectory_opt[..., i], "r+", label=f"Optimal {i+1}")
                axs55[i].plot(times, [-np.pi] * len(times), "m", label=f"Joint Min : {-np.pi:.3f}")
                axs55[i].plot(times, [np.pi] * len(times), "c", label=f"Joint Max : {np.pi:.3f}")
                axs55[i].set_ylabel(f"theta {i+1}")
                axs55[i].set_xlim(times[0], times[-1])
                axs55[i].legend(loc="upper right")
                axs55[i].grid(True)
            axs55[-1].set_xlabel("Time")

    plt.show()
