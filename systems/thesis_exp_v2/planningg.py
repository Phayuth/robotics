import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from spatial_geometry.spatial_transformation import RigidBodyTransformation as rbt
import matplotlib.pyplot as plt
from simulator.sim_planar_rr import RobotArm2DSimulator

# planner
from planner.planner_manipulator import PlannerManipulator
from planner.sampling_based.rrt_plotter import RRTPlotter


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

pm = PlannerManipulator(xStart, xApp, xGoal, plannarConfigDualTreea)

patha = pm.planning()
# fig, ax = plt.subplots()
# ax.set_axis_off()
# fig.set_size_inches(w=3.40067, h=3.40067)
# fig.tight_layout()
# plt.xlim((-np.pi, np.pi))
# plt.ylim((-np.pi, np.pi))
# # RRTPlotter.plot_2d_config_single_tree(pm.planner, patha, ax)
# RRTPlotter.plot_2d_config_dual_tree(pm.planner, patha, ax)

# plt.show()

# fig, ax = plt.subplots()
# RRTPlotter.plot_performance(pm.planner.perfMatrix, ax)
# plt.show()



# thetas = np.hstack((q1, q2, q3)).T
# ax = sim.plot_view(thetas)
ax = sim.plot_view(patha.T)
circle1 = plt.Circle((-3.0, 0.5), rc, color="r")
ax.add_patch(circle1)
plt.show()