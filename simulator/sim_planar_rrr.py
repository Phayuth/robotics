import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
from spatial_geometry.spatial_shape import ShapeRectangle
from robot.nonmobile.planar_rrr import PlanarRRR


class RobotArm2DRRRSimulator:

    def __init__(self):
        # required for planner
        self.configLimit = [[-2*np.pi, 2*np.pi], [-2*np.pi, 2*np.pi], [-2*np.pi, 2*np.pi]]
        self.configDoF = len(self.configLimit)

        self.robot = PlanarRRR()
        xTarg = 1.5
        yTarg = 0.2
        alphaTarg = 2  # given from grapse pose candidate
        hD = 0.25
        wD = 0.25
        rCrop = 0.1
        phiTarg = alphaTarg - np.pi

        xTopStart = (rCrop + hD) * np.cos(alphaTarg - np.pi / 2) + xTarg
        yTopStart = (rCrop + hD) * np.sin(alphaTarg - np.pi / 2) + yTarg
        xBotStart = (rCrop) * np.cos(alphaTarg + np.pi / 2) + xTarg
        yBotStart = (rCrop) * np.sin(alphaTarg + np.pi / 2) + yTarg
        recTop = ShapeRectangle(xTopStart, yTopStart, hD, wD, angle=alphaTarg)
        recBot = ShapeRectangle(xBotStart, yBotStart, hD, wD, angle=alphaTarg)
        self.taskMapObs = [recTop, recBot]

        target = np.array([xTarg, yTarg, phiTarg]).reshape(3, 1)
        thetaGoal = self.inverse_kinematic_geometry(target, elbow_option=0)

    def collision_check(self, xNewConfig):
        raise NotImplementedError

    def get_cspace_grid(self): #generate into 2d array plot by imshow
        raise NotImplementedError

    def plot_taskspace(self, theta):
        raise NotImplementedError

    def plot_cspace(self, axis):
        raise NotImplementedError

    def play_back_path(self, path, axis):
        raise NotImplementedError