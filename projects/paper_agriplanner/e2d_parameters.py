import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
from simulator.sim_planar_rr import RobotArm2DSimulator, NonMobileTaskMap
from spatial_geometry.spatial_transformation import RigidBodyTransformation as rbt
from spatial_geometry.utils import Utils


class PaperLengths:
    latex_linewidth_inch = 3.19423
    latex_columnwidth_inch = 3.19423
    latex_textwidth_inch = 6.68459
    mmtopoint = 72 / 25.4
    inchtopoint = 72


rc = 0.8
cx = -3.0
cy = 0.5
g1 = rbt.conv_polar_to_cartesian(rc, 0.0, cx, cy)
g2 = rbt.conv_polar_to_cartesian(rc, 0.5, cx, cy)  # intended collision
g3 = rbt.conv_polar_to_cartesian(rc, -0.5, cx, cy)

sim = RobotArm2DSimulator()

robot = sim.robot
q0 = np.array([0, 0]).reshape(2, 1)

q1 = robot.inverse_kinematic_geometry(np.array(g1).reshape(2, 1), elbow_option=1)
q2 = robot.inverse_kinematic_geometry(np.array(g2).reshape(2, 1), elbow_option=1)
q3 = robot.inverse_kinematic_geometry(np.array(g3).reshape(2, 1), elbow_option=1)

q1i = robot.inverse_kinematic_geometry(np.array(g1).reshape(2, 1), elbow_option=0)
q2i = robot.inverse_kinematic_geometry(np.array(g2).reshape(2, 1), elbow_option=0)
q3i = robot.inverse_kinematic_geometry(np.array(g3).reshape(2, 1), elbow_option=0)

q1i = Utils.wrap_to_pi(q1i)
q2i = Utils.wrap_to_pi(q2i)
q3i = Utils.wrap_to_pi(q3i)


xStart = q0
xApp = [q1, q2, q3, q1i, q2i, q3i]
xGoal = [q1, q2, q3, q1i, q2i, q3i]

thetas = np.hstack((q0, q1, q2, q3, q1i, q2i, q3i))
