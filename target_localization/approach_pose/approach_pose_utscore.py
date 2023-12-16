import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from geometry.geometry_class import CoordinateTransform

# SECTION - target define pose
x_targ = 1.5
y_targ = 0.5
t_targ = np.linspace(np.pi / 2, 3 * np.pi / 2, 50)  # Create canidate pose


# SECTION - calculate inner and outer pose for circle
r_inner = 0.1
r_outer = r_inner + 0.1
x_inner, y_inner = CoordinateTransform.polar_to_cartesian(r_inner, t_targ, x_targ, y_targ)
x_outer, y_outer = CoordinateTransform.polar_to_cartesian(r_outer, t_targ, x_targ, y_targ)


# SECTION - calculate score of each angle
score_candidate = []
w1 = 0.9  # weight on angle
w2 = 0.1  # weight on distant
w3 = 0

for i in range(len(x_inner)):

    # score on angle
    x_in = x_inner[i]
    y_in = y_inner[i]
    a = [x_in - x_targ, y_in - y_targ]
    b = [1, 0]
    dot = np.dot(a, b)
    alpha_candidate = np.arccos(dot / (np.linalg.norm(a) * np.linalg.norm(b)))
    s_alpha = 1 / (np.pi - alpha_candidate)

    # score on distant from origin
    s_dist = np.linalg.norm([x_in - 0, y_in - 0])

    # score on distant on distribution
    s_div = 0

    # find total score process
    total_norm = np.linalg.norm([s_alpha, s_dist, s_div])
    s_alpha_norml = s_alpha / total_norm
    s_dist_norml = s_dist / total_norm
    s_div_norml = s_div / total_norm

    util_score = w1 * s_alpha_norml + w2 * s_dist_norml + w3 * s_div_norml
    score_candidate.append(util_score)

score_candidate = np.array(score_candidate)
print("==>> sc: \n", score_candidate)

score_candidate_sorted = np.sort(score_candidate, axis=None)
print(f"==>> score_candidate_sorted: \n{score_candidate_sorted}")

maxim = np.argmax(score_candidate)  # find the maximum score
print("==>> maxim: \n", maxim)

maxim_sorted = np.argmax(score_candidate)  # find the maximum score
print("==>> maxim_sorted: \n", maxim_sorted)


# SECTION - plot pose angle
plt.axes().set_aspect('equal')
plt.axvline(x=0, c="black")
plt.axhline(y=0, c="black")
# circle_plt(x_targ, y_targ, r_inner)

for i in range(len(x_inner)):
    plt.plot([x_inner[i], x_outer[i]], [y_inner[i], y_outer[i]], c="orange")
plt.plot([x_inner[maxim], x_outer[maxim]], [y_inner[maxim], y_outer[maxim]], c="cyan")  # plot the choosen angle
plt.plot([x_inner[maxim_sorted], x_outer[maxim_sorted]], [y_inner[maxim_sorted], y_outer[maxim_sorted]], c="red")  # plot the choosen angle

plt.show()
