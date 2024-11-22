import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from chomp_cfg import cfg

np.set_printoptions(linewidth=2000)

# def compute_smooth_loss(xi, start, end, diff_matrix, time_interval):
#     """
#     Computes smoothness loss and gradient for a trajectory.

#     Parameters:
#     - xi: Trajectory matrix (shape: n x d)
#     - start: Start configuration (shape: d)
#     - end: End configuration (shape: d)
#     - diff_matrix: Finite difference matrix (shape: (n-1) x n)
#     - time_interval: Time step (scalar)

#     Returns:
#     - smoothness_loss: Scalar value representing smoothness cost
#     - smoothness_grad: Gradient of the smoothness loss (shape: n x d)
#     """

#     # Precompute boundary effects
#     n, d = xi.shape
#     ed = np.zeros([n + 1, d])
#     ed[0] = start / time_interval  # Start boundary condition
#     ed[-1] = end / time_interval  # End boundary condition

#     # Compute the velocity using finite difference matrix
#     velocity = diff_matrix.dot(xi)
#     print(f"> velocity: {velocity}")

#     # Compute the velocity norm with boundary correction
#     velocity_norm = np.linalg.norm(velocity + ed[1:-1], axis=1)
#     print(f"> velocity_norm: {velocity_norm}")

#     # Compute the smoothness loss as 1/2 * ||velocity||^2
#     smoothness_loss = 0.5 * np.sum(velocity_norm**2)
#     print(f"> smoothness_loss: {smoothness_loss}")

#     # Compute the gradient of the smoothness loss
#     smoothness_grad = diff_matrix.T.dot(velocity)
#     print(f"> smoothness_grad: {smoothness_grad}")

#     return smoothness_loss, smoothness_grad


def compute_smooth_like_paper():
    xi = np.array([[0.0, 0.0], [3.0, 2.0], [5.0, 5.0], [9.0, 7.0]])
    dt = 1.0

    diff = np.diff(xi, axis=0) / dt
    diffsq = diff**2
    segsum = np.sum(diffsq, axis=1)
    fsmooth = 0.5 * np.sum(segsum)
    print(f"> fsmooth: {fsmooth}")

    K = np.array([[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]])


if __name__ == "__main__":

    # # Finite difference matrix (for 4 waypoints)
    # diff_matrix = np.array([
    #     [-1, 1, 0, 0],
    #     [0, -1, 1, 0],
    #     [0, 0, -1, 1]
    # ])

    # # Time interval between waypoints
    # time_interval = 1.0

    # # Compute the smoothness loss and gradient
    # smoothness_loss, smoothness_grad = compute_smooth_loss(xi, start, end, diff_matrix, time_interval)

    # # Output the results
    # print(f"Smoothness Loss: {smoothness_loss}")
    # print(f"Smoothness Gradient:\n{smoothness_grad}")

    # compute_smooth_like_paper()

    def get_diff_matrix_K(n, diff_rules, time_interval, diff_rule_length=7, order=1, with_end=True):
        diff_rule = diff_rules[order - 1]
        half_length = diff_rule_length // 2
        diff_matrix = np.zeros([n + 1, n])
        for i in range(0, n + 1):
            for j in range(-half_length, half_length):
                index = i + j
                if index >= 0 and index < n:
                    diff_matrix[i, index] = diff_rule[j + half_length]
        if with_end == False:
            diff_matrix[-1, -1] = 0
        return diff_matrix / (time_interval**order)

    cfg.timesteps = 4  # discretize uniform timesteps for trajectory
    cfg.time_interval = (0.1 * cfg.timesteps) / cfg.timesteps

    cfg.diff_rule_length = 7
    cfg.diff_rule = np.array([[0, 0, -1, 1, 0, 0, 0], [0, 0, 1, -2, 1, 0, 0], [0, -0.5, 1, 0, -1, 0.5, 0]])
    # cfg.diff_matrices = [get_diff_matrix_K(cfg.timesteps, cfg.diff_rule, cfg.time_interval, cfg.diff_rule_length, i + 1, not cfg.goal_set_proj) for i in range(cfg.diff_rule.shape[0])]
    cfg.diff_matrices = [get_diff_matrix_K(cfg.timesteps, cfg.diff_rule, cfg.time_interval, cfg.diff_rule_length, i + 1, True) for i in range(cfg.diff_rule.shape[0])]

    print(f"> cfg.diff_matrices:\n {cfg.diff_matrices[0]}")
    print(f"> cfg.diff_matrices.shape: {cfg.diff_matrices[0].shape}")

    cfg.A = cfg.diff_matrices[0].T.dot(cfg.diff_matrices[0])
    print(f"> cfg.A: {cfg.A}")
    print(f"> cfg.A.shape: {cfg.A.shape}")
    cfg.Ainv = np.linalg.inv(cfg.A)

    """
    Computes smoothness loss
    """
    # A typical 9-link robot could include a 7-joint manipulator (like a robotic arm with 7 degrees of freedom) and possibly additional components such as a gripper or base links.
    # K matrix should be n+1 x n
    d = 9  # it seem we have to plan in 9 joints/links
    xi = np.random.uniform(-np.pi, np.pi, size=(cfg.timesteps, d))
    start = np.random.uniform(-np.pi, np.pi, size=(1, d))
    end = np.random.uniform(-np.pi, np.pi, size=(1, d))

    link_smooth_weight = np.array(cfg.link_smooth_weight)[None]  # add additional dimension from (9,) to (1,9)
    print(f"> link_smooth_weight: {link_smooth_weight}")
    print(f"> link_smooth_weight.shape: {link_smooth_weight.shape}")

    ed = np.zeros([xi.shape[0] + 1, xi.shape[1]])
    ed[0] = cfg.diff_rule[0][cfg.diff_rule_length // 2 - 1] * start / cfg.time_interval
    if not cfg.goal_set_proj:
        ed[-1] = cfg.diff_rule[0][cfg.diff_rule_length // 2] * end / cfg.time_interval
    print(f"> ed: {ed}")
    print(f"> ed.shape: {ed.shape}")

    velocity = cfg.diff_matrices[0].dot(xi)
    print(f"> velocity: {velocity}")
    print(f"> velocity.shape: {velocity.shape}")

    velocity_norm = np.linalg.norm((velocity + ed) * link_smooth_weight, axis=1)
    print(f"> velocity_norm: {velocity_norm}")
    print(f"> velocity_norm.shape: {velocity_norm.shape}")

    smoothness_loss = 0.5 * velocity_norm**2
    print(f"> smoothness_loss: {smoothness_loss}")
    print(f"> smoothness_loss.shape: {smoothness_loss.shape}")

    smoothness_grad = cfg.A.dot(xi) + cfg.diff_matrices[0].T.dot(ed)
    smoothness_grad *= link_smooth_weight
    print(f"> smoothness_grad: {smoothness_grad}")
    print(f"> smoothness_grad.shape: {smoothness_grad.shape}")

    smoothness_loss_sum = smoothness_loss.sum()
    print(f"> smoothness_loss_sum: {smoothness_loss_sum}")
    print(f"> smoothness_loss_sum.shape: {smoothness_loss_sum.shape}")

    step = 1  # integer increasing by 1 everytime we update
    cfg.smoothness_weight = cfg.smoothness_base_weight * cfg.cost_schedule_boost**step
    print(f"> cfg.smoothness_weight: {cfg.smoothness_weight}")

    weighted_smooth = cfg.smoothness_weight * smoothness_loss_sum
    print(f"> weighted_smooth: {weighted_smooth}")
    print(f"> weighted_smooth.shape: {weighted_smooth.shape}")



    # cost = weighted_obs + weighted_smooth

