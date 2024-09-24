import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
import matplotlib.pyplot as plt

from spatial_geometry.spatial_transformation import RigidBodyTransformation as rbt
from simulator.sim_planar_rr import RobotArm2DSimulator
from planner.sampling_based.rrt_plannerapi import RRTPlannerAPI

rc = 0.8
g1 = rbt.conv_polar_to_cartesian(rc, 0.0, -3.0, 0.5)
g2 = rbt.conv_polar_to_cartesian(rc, 0.5, -3.0, 0.5)  # intended collision
g3 = rbt.conv_polar_to_cartesian(rc, -0.5, -3.0, 0.5)

sim = RobotArm2DSimulator()

robot = sim.robot
q0 = np.array([0, 0]).reshape(2, 1)
q1 = robot.inverse_kinematic_geometry(np.array(g1).reshape(2, 1), elbow_option=1)
q2 = robot.inverse_kinematic_geometry(np.array(g2).reshape(2, 1), elbow_option=1)
q3 = robot.inverse_kinematic_geometry(np.array(g3).reshape(2, 1), elbow_option=1)


thetas = np.hstack((q0, q1, q2, q3)).T
ax = sim.plot_view(thetas)
circle1 = plt.Circle((-3.0, 0.5), rc, color="r")
ax.add_patch(circle1)
plt.show()


plannarConfigDualTreea = {
    "planner": 5,
    "eta": 0.3,
    "subEta": 0.05,
    "maxIteration": 2000,
    "simulator": sim,
    "nearGoalRadius": None,
    "rewireRadius": None,
    "endIterationID": 1,
    "printDebug": True,
    "localOptEnable": True,
}

xStart = np.array([0, 0]).reshape(2, 1)
xApp = q1
xGoal = q1


# xStart = np.array([0, 0]).reshape(2, 1)
# xApp = [q1, q2, q3]
# xGoal = [q1, q2, q3]

pm = RRTPlannerAPI.init_normal(xStart, xApp, xGoal, plannarConfigDualTreea)
patha = pm.begin_planner()

pm.plot_performance()

ax = sim.plot_view(patha.T)
circle1 = plt.Circle((-3.0, 0.5), rc, color="r")
ax.add_patch(circle1)
plt.show()