import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
from spatial_geometry.utils import Utils
from simulator.sim_planar_rrr import RobotArm2DRRRSimulator, NonMobileTaskMap


class PaperLengths:
    latex_linewidth_inch = 3.19423
    latex_columnwidth_inch = 3.19423
    latex_textwidth_inch = 6.68459
    mmtopoint = 72 / 25.4
    inchtopoint = 72


rc = 0.8
cx = -3.0
cy = 0.5
g1 = np.array([cx, cy, 0.0 - np.pi]).reshape(3, 1)
g2 = np.array([cx, cy, 0.5 - np.pi]).reshape(3, 1)  # intended collision
g3 = np.array([cx, cy, -0.5 - np.pi]).reshape(3, 1)

sim = RobotArm2DRRRSimulator()

robot = sim.robot
q0 = np.array([0, 0, 0]).reshape(3, 1)

q1 = robot.inverse_kinematic_geometry(g1, elbow_option=1)
q2 = robot.inverse_kinematic_geometry(g2, elbow_option=1)
q3 = robot.inverse_kinematic_geometry(g3, elbow_option=1)

q1i = robot.inverse_kinematic_geometry(g1, elbow_option=0)
q2i = robot.inverse_kinematic_geometry(g2, elbow_option=0)
q3i = robot.inverse_kinematic_geometry(g3, elbow_option=0)

q1i = Utils.wrap_to_pi(q1i)
q2i = Utils.wrap_to_pi(q2i)
q3i = Utils.wrap_to_pi(q3i)

xStart = q0
xApp = [q1, q3, q1i, q2i, q3i]
xGoal = [q1, q3, q1i, q2i, q3i]

thetas = np.hstack((q0, q1, q2, q3, q1i, q2i, q3i))
