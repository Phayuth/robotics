import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stomp import STOMP, STOMPUTILS
from matplotlib.patches import Polygon as pltPolygon
from shapely.geometry import Point, Polygon, MultiPolygon, LineString
import concurrent.futures

# TODO : fix python loop at shapely arm collision, it is insanely slow
np.set_printoptions(linewidth=1000, suppress=True)
np.random.seed(9)


def forward(theta):
    l1 = 1
    l2 = 1
    l3 = 1
    l4 = 1
    l5 = 1
    l6 = 1
    t1, t2, t3, t4, t5, t6 = theta

    A = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1],
        ]
    )
    b = np.array(
        [
            [0, 0],
            [l1 * np.cos(t1), l1 * np.sin(t1)],
            [l2 * np.cos(t1 + t2), l2 * np.sin(t1 + t2)],
            [l3 * np.cos(t1 + t2 + t3), l3 * np.sin(t1 + t2 + t3)],
            [l4 * np.cos(t1 + t2 + t3 + t4), l4 * np.sin(t1 + t2 + t3 + t4)],
            [l5 * np.cos(t1 + t2 + t3 + t4 + t5), l5 * np.sin(t1 + t2 + t3 + t4 + t5)],
            [l6 * np.cos(t1 + t2 + t3 + t4 + t5 + t6), l6 * np.sin(t1 + t2 + t3 + t4 + t5 + t6)],
        ]
    )

    links = A @ b
    return links


# theta = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
# links = forward(theta)
# print(f"> links.shape: {links.shape}")
# line = LineString(links)
# armcol = line.buffer(0.3)

# obstacles
o1 = Polygon([(3, 3), (4, 3), (4, 4), (3, 4)])
o2 = Polygon([(-4, 3), (-3, 3), (-3, 4), (-4, 4)])
oo = MultiPolygon([o1, o2])
clearence = 0.1
rb = 0.3
# t = armcol.intersects(oo)
# d = armcol.distance(oo)

# polygon = pltPolygon(armcol.exterior, facecolor="none", edgecolor="r")
# fig, ax = plt.subplots()
# ax.plot(links[:, 0], links[:, 1], "ro")
# ax.add_patch(polygon)
# for o in oo:
#     polygon = pltPolygon(o.exterior, facecolor="none", edgecolor="r")
#     ax.add_patch(polygon)
# ax.set_xlim(-7, 7)
# ax.set_ylim(-7, 7)
# ax.set_aspect("equal")
# # plt.show()


def forward_kinematics_vectorized(joint_angles):
    link_lengths = np.ones(6)  # Assuming all link lengths are 1 unit
    n, num_joints = joint_angles.shape
    cumulative_angles = np.cumsum(joint_angles, axis=1)
    x_displacements = link_lengths * np.cos(cumulative_angles)
    y_displacements = link_lengths * np.sin(cumulative_angles)
    displacements = np.stack((x_displacements, y_displacements), axis=-1)
    positions = np.cumsum(displacements, axis=1)
    base = np.zeros((n, 1, 2))  # Shape (n, 1, 2), to represent the base at (0, 0)
    result = np.concatenate([base, positions], axis=1)  # Shape (n, 7, 2)
    return result


def get_dmin(p):
    dmin = LineString(p).buffer(rb).distance(oo)
    return dmin

# my parallel version is not working well yet
def compute_obstacle_cost_one_traj(traj, use_parallel=False):
    xy_coordinates = forward_kinematics_vectorized(traj)
    if use_parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:  # finished in order
            results = executor.map(get_dmin, xy_coordinates)
            dmin = np.array(list(results))
    else:
        armcols = [LineString(pp).buffer(rb) for pp in xy_coordinates]
        dmin = np.array([armcol.distance(oo) for armcol in armcols])
    cost = np.maximum(clearence + rb - dmin, 0)
    return np.sum(cost)


def compute_obstacle_cost_one_traj_to_each_joint(traj, use_parallel=False):
    # input noisy trajectory K, N, d
    xy_coordinates = forward_kinematics_vectorized(traj)
    if use_parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:  # finished in order
            results = executor.map(get_dmin, xy_coordinates)
            dmin = np.array(list(results))
    else:
        armcols = [LineString(pp).buffer(rb) for pp in xy_coordinates]
        dmin = np.array([armcol.distance(oo) for armcol in armcols])
    cost = np.maximum(clearence + rb - dmin, 0)
    costfully = np.repeat(cost[:, np.newaxis], d, axis=1)
    return costfully


def compute_obstacle_cost_one_traj_to_each_joint_k_traj(noisy_traj):
    costfully = np.zeros(shape=noisy_traj.shape)

    for i in range(noisy_traj.shape[0]):
        cost = compute_obstacle_cost_one_traj_to_each_joint(noisy_traj[i])
        costfully[i] = cost

    return costfully


# sys.exit(0)
if __name__ == "__main__":
    K = 50
    N = 100
    d = 6
    h = 10
    tolerance = 0.1
    Rscale = 100
    stddev = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    num_iterations = 500

    # environment
    theta_s = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    theta_g = np.array([np.pi, 0.0, 0.0, 0.0, 0.0, 0.0])
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
            # axs55[i].legend(loc="upper right")
            axs55[i].grid(axis="x")
        axs55[-1].set_xlabel("Time")

    # task space plot animation
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_aspect("equal")
    for o in oo:
        polygon = pltPolygon(o.exterior, facecolor="r", edgecolor="r", alpha=0.5)
        ax.add_patch(polygon)
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)

    (robotLinks,) = ax.plot([], [], "b")
    c = [plt.Circle([0, 0], rb, color="b", alpha=0.5, fill=False) for i in range(7)]

    def update(frame):
        pToB = forward(trajectory_opt[frame]).T
        robotLinks.set_data(pToB[0, ...], pToB[1, ...])

        for i, ci in enumerate(c):
            ci.set_center(pToB[:, i])
            ax.add_patch(ci)

    animation = animation.FuncAnimation(fig, update, frames=(trajectory_opt.shape[0]), interval=100)
    plt.show()
