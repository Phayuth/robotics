import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from icecream import ic
from stomp import STOMP, STOMPUTILS

np.set_printoptions(linewidth=1000, suppress=True)


def hrz(theta):
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def ht(x, y, z):
    return np.array(
        [
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ]
    )


# obstacle
cobs = np.array([0.0, 1.5]).reshape(2, 1)  # obs center
robs = 0.2  # robs
rb = 1 / 6  # rbodies
clearence = 0.1  # clearence distance


def forward_points(thetas):
    l1 = 1
    l2 = 1
    Hj1ToB = hrz(thetas[0])
    Hj2Toj1 = ht(l1, 0, 0) @ hrz(thetas[1])

    P0123ToJ1 = np.array(
        [
            [0, l1 / 3, 2 * l1 / 3, l1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ]
    )
    P456ToJ2 = np.array(
        [
            [l2 / 3, 2 * l2 / 3, l2],
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
        ]
    )

    P123ToB = Hj1ToB @ P0123ToJ1
    P456ToB = Hj1ToB @ Hj2Toj1 @ P456ToJ2

    bodies_points = np.hstack((P123ToB[0:2, ...], P456ToB[0:2, ...]))
    return bodies_points


def compute_sdf(bodies_points):
    ec = bodies_points.T - cobs.T
    dmin = np.linalg.norm(ec, axis=1) - robs - rb
    return dmin


def compute_obstacle_cost(bodies_points):
    dmin = compute_sdf(bodies_points)
    cost = np.maximum(clearence + rb - dmin, 0)
    return np.sum(cost)


# thetas = np.array([np.pi / 2, 0])
# bodies_points = forward_points(thetas)
# plt.plot(bodies_points[0, ...], bodies_points[1, ...], "b+")
# circle1 = plt.Circle(cobs, robs, color="r", alpha=0.5)
# plt.gca().add_patch(circle1)
# for i in range(bodies_points.shape[1]):
#     circlebody = plt.Circle(bodies_points[:, i], 1 / 6, color="g", alpha=0.5)
#     plt.gca().add_patch(circlebody)
# plt.xlim(-3, 3)
# plt.ylim(-3, 3)
# plt.gca().set_aspect("equal", adjustable="box")
# plt.show()

# ====================START HERE====================


def gen_trajectory():
    n = 10  # nwaypoints
    d = 2  # 2 joints
    start = np.array([0, 0])
    end = np.array([np.pi, 0])

    xi = np.linspace(start, end, n, endpoint=True)
    return xi


def hrz_array(theta):
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    hzz = np.array(
        [
            [cos_t, -sin_t, np.zeros_like(theta), np.zeros_like(theta)],
            [sin_t, cos_t, np.zeros_like(theta), np.zeros_like(theta)],
            [np.zeros_like(theta), np.zeros_like(theta), np.ones_like(theta), np.zeros_like(theta)],
            [np.zeros_like(theta), np.zeros_like(theta), np.zeros_like(theta), np.ones_like(theta)],
        ]
    )

    hzz = np.transpose(hzz, (2, 0, 1))  # Transpose the result to get shape (n, 4, 4)
    return hzz


def ht_array(x, y, z):
    htt = np.array(
        [
            [np.ones_like(x), np.zeros_like(x), np.zeros_like(x), x],
            [np.zeros_like(x), np.ones_like(x), np.zeros_like(x), y],
            [np.zeros_like(x), np.zeros_like(x), np.ones_like(x), z],
            [np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), np.ones_like(x)],
        ]
    )

    htt = np.transpose(htt, (2, 0, 1))
    return htt


def forward_points_array(thetas):
    l1 = 1
    l2 = 1
    Hj1ToB = hrz_array(thetas[:, 0])
    Hj2Toj1 = ht_array(np.full(thetas.shape[0], l1), np.zeros(thetas.shape[0]), np.zeros(thetas.shape[0])) @ hrz_array(thetas[:, 1])

    P0123ToJ1 = np.array(
        [
            [0, l1 / 3, 2 * l1 / 3, l1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ]
    )
    P456ToJ2 = np.array(
        [
            [l2 / 3, 2 * l2 / 3, l2],
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
        ]
    )

    P123ToB = Hj1ToB @ P0123ToJ1
    P456ToB = Hj1ToB @ Hj2Toj1 @ P456ToJ2

    pToB = np.concatenate((P123ToB, P456ToB), axis=2)
    pToB = pToB[:, 0:2, :]
    return pToB


def compute_obstacle_cost_one_traj(traj):
    bodies_points_trajectory = forward_points_array(traj).transpose(0, 2, 1)
    ec = bodies_points_trajectory - cobs.flatten()
    dmin = np.linalg.norm(ec, axis=2, keepdims=True) - robs - rb
    m = np.maximum(clearence + rb - dmin, 0)
    cost = np.sum(m, axis=1).reshape(-1)
    return np.sum(cost)


def compute_obstacle_cost_one_traj_to_each_joint(traj):
    bodies_points_trajectory = forward_points_array(traj).transpose(0, 2, 1)
    ec = bodies_points_trajectory - cobs.flatten()
    dmin = np.linalg.norm(ec, axis=2, keepdims=True) - robs - rb
    m = np.maximum(clearence + rb - dmin, 0)
    cost = np.sum(m, axis=1).reshape(-1)
    costfully = np.repeat(cost[:, np.newaxis], d, axis=1)
    return costfully


def compute_obstacle_cost_one_traj_to_each_joint_k_traj(noisy_traj):
    costfully = np.zeros(shape=noisy_traj.shape)

    for i in range(noisy_traj.shape[0]):
        cost = compute_obstacle_cost_one_traj_to_each_joint(noisy_traj[i])
        costfully[i] = cost

    return costfully


# traj = gen_trajectory()
# cost = compute_obstacle_cost_one_traj(traj)
# ic(cost)

# cost = compute_obstacle_cost_one_traj_to_each_joint(traj)
# ic(cost)
# ic(cost.shape)

if __name__ == "__main__":
    K = 50
    N = 100
    d = 2
    h = 10
    tolerance = 0.1
    Rscale = 100
    stddev = [10.0, 10.0]
    num_iterations = 500

    # environment
    theta_s = np.array([0.0, 0.0])
    theta_g = np.array([np.pi, np.pi / 2])
    times = np.linspace(0, N, num=N)

    trajectory = STOMPUTILS.generate_lerp_trajectory(theta_s, theta_g, N)
    stomp = STOMP(
        K=K,
        N=N,
        d=d,
        h=h,
        tolerance=tolerance,
        Rscale=Rscale,
        stddev=stddev,
        num_iterations=num_iterations,
        func_obstacle_cost=compute_obstacle_cost_one_traj_to_each_joint_k_traj,
        func_constraint_cost=STOMPUTILS.constraint_cost,
        func_torque_cost=STOMPUTILS.torque_cost,
        func_obstacle_cost_single=compute_obstacle_cost_one_traj,
        func_constraint_cost_single=STOMPUTILS.compute_trajectory_constraint_cost,
        func_torque_cost_single=STOMPUTILS.compute_trajectory_torque_cost,
        seed=9,
        print_debug=False,
        print_log=False,
    )
    trajectory_opt = stomp.optimize(trajectory)

    # plot comparison
    if True:
        fig55, axs55 = plt.subplots(d, 1, sharex=True)
        for i in range(d):
            axs55[i].plot(times, trajectory[..., i], "k--", label=f"Original {i+1}")
            axs55[i].plot(times, trajectory_opt[..., i], "r+", label=f"Optimal {i+1}")
            axs55[i].plot(times, [-2 * np.pi] * len(times), "m", label=f"Joint Min : {-2*np.pi:.3f}")
            axs55[i].plot(times, [2 * np.pi] * len(times), "c", label=f"Joint Max : {2*np.pi:.3f}")
            axs55[i].plot(times, [-np.pi] * len(times), "m", label=f"Joint : {-np.pi:.3f}")
            axs55[i].plot(times, [np.pi] * len(times), "c", label=f"Joint : {np.pi:.3f}")
            axs55[i].plot(times, [0] * len(times), "k")
            axs55[i].set_ylabel(f"theta {i+1}")
            axs55[i].set_xlim(times[0], times[-1])
            axs55[i].legend(loc="upper right")
            axs55[i].grid(axis="x")
        axs55[-1].set_xlabel("Time")

    # task space plot animation
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_aspect("equal")
    circle1 = plt.Circle(cobs, robs, color="r")
    ax.add_patch(circle1)
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    (robotLinks,) = ax.plot([], [], "b")
    c = [plt.Circle([0, 0], rb, color="b", alpha=0.5, fill=False) for i in range(7)]

    def update(frame):
        pToB = forward_points(trajectory_opt[frame])
        robotLinks.set_data(pToB[0, ...], pToB[1, ...])

        for i, ci in enumerate(c):
            ci.set_center(pToB[:, i])
            ax.add_patch(ci)

    animation = animation.FuncAnimation(fig, update, frames=(trajectory_opt.shape[0]), interval=100)
    plt.show()
