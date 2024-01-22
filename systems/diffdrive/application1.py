"""
Application 1
Robot : DiffDrive
Planner : RRT*
Controller : Path PurePursuit
"""
import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from matplotlib import animation
np.random.seed(1010)
from controllers.differential.path_purepursuit import DifferentialDrivePurePursuitController
from planner.planner_mobilerobot import PlannerMobileRobot
from simulator.sim_diffdrive import DiffDrive2DSimulator
from simulator.integrator_euler import EulerNumericalIntegrator

# robot and env obs
qStart = np.array([29.50, 6.10]).reshape(2,1)
qAux = np.array([15.00, 2.80]).reshape(2,1)
qGoal = np.array([15.90, 2.30]).reshape(2,1)

env = DiffDrive2DSimulator()

# planner
plannarConfigDualTreea = {
    "planner": 1,
    "eta": 0.3,
    "subEta": 0.05,
    "maxIteration": 3000,
    "simulator": env,
    "nearGoalRadius": None,
    "rewireRadius": None,
    "endIterationID": 1,
    "printDebug": True,
    "localOptEnable": True
}

planner = PlannerMobileRobot(qStart, qAux, qGoal, plannarConfigDualTreea)
path = planner.planning()
path = path.T

# controller
controller = DifferentialDrivePurePursuitController(path, loopMode=False)

# simulator
def dynamic(currentPose, input):
    return env.robot.forward_external_kinematic(input, currentPose[2,0])

def desired(currentPose, time):
    return np.array([0.0, 0.0, 0]).reshape(3, 1)

def control(currentPose, desiredPose):
    return controller.kinematic_control(currentPose)

q0 = np.array([29.50, 6.10, 0.0]).reshape(3, 1)
tSpan = (0, 132)
dt = 0.01
intg = EulerNumericalIntegrator(dynamic, control, desired, q0, tSpan, dt)
timeSteps, states, desireds, controls = intg.simulation()

env.play_back_path(states, animation)