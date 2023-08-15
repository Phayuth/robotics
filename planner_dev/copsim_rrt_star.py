""" 
Devplanner with RRT Star with reject goal sampling

"""
import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
import numpy as np
from planner_dev.copsim_rrt_component import Node, CopSimRRTComponent


class RRTStarDev(CopSimRRTComponent):
    def __init__(self, xStart, xApp, xGoal, eta=0.3, maxIteration=1000) -> None:
        super().__init__()
        # start, aux, goal node
        self.xStart = Node(xStart[0, 0], xStart[1, 0], xStart[2, 0], xStart[3, 0], xStart[4, 0], xStart[5, 0])
        self.xGoal = Node(xGoal[0, 0], xGoal[1, 0], xGoal[2, 0], xGoal[3, 0], xGoal[4, 0], xGoal[5, 0])
        self.xApp = Node(xApp[0, 0], xApp[1, 0], xApp[2, 0], xApp[3, 0], xApp[4, 0], xApp[5, 0])

        self.eta = eta
        self.maxIteration = maxIteration
        self.treeVertex = [self.xStart]
        self.distGoalToApp = self.distance_between_config(self.xGoal, self.xApp)
        self.rewireRadius = 0.5
        self.XInGoalRegion = []


    def planning(self):
        self.copHandle.start_sim()

        timePlanningStart = time.perf_counter_ns()
        itera = self.planner_rrt_star()
        path = self.search_singledirection_path()
        timePlanningEnd = time.perf_counter_ns()

        # record performance
        self.perfMatrix["Planner Name"] = "RRT Star"
        self.perfMatrix["Parameters"]["eta"] = self.eta 
        self.perfMatrix["Parameters"]["Max Iteration"] = self.maxIteration
        self.perfMatrix["Parameters"]["Rewire Radius"] = self.rewireRadius
        self.perfMatrix["Number of Node"] = len(self.treeVertex)
        self.perfMatrix["Total Planning Time"] = (timePlanningEnd-timePlanningStart) * 1e-9
        self.perfMatrix["Planning Time Only"] = self.perfMatrix["Total Planning Time"] - self.perfMatrix["KCD Time Spend"]* 1e-9
        self.perfMatrix["Average KCD Time"] = self.perfMatrix["KCD Time Spend"] / self.perfMatrix["Number of Collision Check"]

        self.copHandle.stop_sim()
        return path
    
    def planner_rrt_star(self):
        for itera in range(self.maxIteration):
            print(itera)
            # this have nothing to do with planning itself, just for record performance data only
            if len(self.XInGoalRegion) == 0:
                cBest = np.inf
                cBestPrevious = np.inf
            else:
                xSolnCost = [xSoln.parent.cost + self.cost_line(xSoln.parent, xSoln) + self.cost_line(xSoln, self.xApp) for xSoln in self.XInGoalRegion]
                print(f"==>> xSolnCost: \n{xSolnCost}")
                cBest = min(xSolnCost)
                if cBest < cBestPrevious :
                    self.perfMatrix["Cost Graph"].append((itera, cBest))
                    cBestPrevious = cBest
            
            xRand = self.uni_sampling() #self.bias_sampling(self.xApp)
            xNearest = self.nearest_node(self.treeVertex, xRand)
            xNew = self.steer(xNearest, xRand, self.eta)
            xNew.parent = xNearest
            if self.is_config_in_region_of_config(xNew, self.xGoal, self.distGoalToApp) or \
                self.is_connect_config_in_region_of_config(xNearest, xNew, self.xGoal, self.distGoalToApp):
                continue
            xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
            if self.is_config_in_collision(xNew) or self.is_connect_config_in_collision(xNew.parent, xNew):
                continue
            else:
                XNear = self.near(self.treeVertex, xNew, self.rewireRadius)
                xMin = xNew.parent
                cMin = xMin.cost + self.cost_line(xMin, xNew)
                for xNear in XNear:
                    if self.is_connect_config_in_collision(xNear, xNew):
                        continue

                    cNew = xNear.cost + self.cost_line(xNear, xNew)
                    if cNew < cMin:
                        xMin = xNear
                        cMin = cNew

                xNew.parent = xMin
                xNew.cost = cMin
                self.treeVertex.append(xNew)

                for xNear in XNear:
                    if self.is_connect_config_in_collision(xNear, xNew):
                        continue
                    cNear = xNear.cost
                    cNew = xNew.cost + self.cost_line(xNew, xNear)
                    if cNew < cNear:
                        xNear.parent = xNew
                        xNear.cost = xNew.cost + self.cost_line(xNew, xNear)

                # in goal region, this have nothing to do with planning itself, just for record performance data only
                if self.is_config_in_region_of_config(xNew, self.xApp, radius=self.eta): # radius = 1
                    self.XInGoalRegion.append(xNew)

        return itera

    def search_singledirection_path(self):
        XNear = self.near(self.treeVertex, self.xApp, self.rewireRadius)
        # for xNear in XNear:
        for xNear in self.treeVertex:
            if self.is_connect_config_in_collision(xNear, self.xApp):
                continue
            self.xApp.parent = xNear

            path = [self.xApp]
            currentNode = self.xApp

            while currentNode != self.xStart:
                currentNode = currentNode.parent
                path.append(currentNode)

            path.reverse()
            bestPath = path
            cost = sum(i.cost for i in path)

            if cost < sum(j.cost for j in bestPath):
                bestPath = path

        return bestPath + [self.xGoal]
    
if __name__ == "__main__":
    np.random.seed(9)
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from planner_util.extract_path_class import extract_path_class_6d
    from target_localization.pre_record_value import thetaInit, thetaGoal1, thetaApp1, qCurrent, qAux, qGoal, wrap_to_pi
    from copsim.arm_api import UR5eStateArmCoppeliaSimAPI
    from util.general_util import write_dict_to_file

    # Define pose
    thetaInit = thetaInit
    thetaGoal = wrap_to_pi(qGoal)
    thetaApp = wrap_to_pi(qAux)
    
    planner = RRTStarDev(thetaInit, thetaApp, thetaGoal, eta=0.3, maxIteration=10000)
    path = planner.planning()
    write_dict_to_file(planner.perfMatrix, "./planner_dev/result_6d/result_6d_rrtstar.txt")
    print(f"==>> path: \n{path}")

    time.sleep(3)

    # play back
    planner.copHandle.start_sim()
    pathX, pathY, pathZ, pathP, pathQ, pathR = extract_path_class_6d(path)
    pathX = np.array(pathX)
    pathY = np.array(pathY)
    pathZ = np.array(pathZ)
    pathP = np.array(pathP)
    pathQ = np.array(pathQ)
    pathR = np.array(pathR)

    armState = UR5eStateArmCoppeliaSimAPI()
    armState.set_goal_joint_value(thetaGoal)
    armState.set_aux_joint_value(thetaApp)
    armState.set_start_joint_value(thetaInit)
    for i in range(len(pathX)):
        jointVal = np.array([pathX[i], pathY[i], pathZ[i], pathP[i], pathQ[i], pathR[i]]).reshape(6, 1)
        planner.copHandle.set_joint_value(jointVal)
        time.sleep(2)
        # triggers next simulation step
        # client.step()

    # stop simulation
    planner.copHandle.stop_sim()

    # Optimization stage, I want to fit the current theta to time and use that information to inform optimization
    time = np.linspace(0, 1, len(pathX))

    def quintic5deg(x, a, b, c, d, e, f):
        return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e*x*f
    
    def polynomial9deg(x, a, b, c, d, e, f, g, h, i):
        return a*x**9 + b*x**8 + c*x**7 + d*x**6 + e*x**5 + f*x**4 + g*x**3 + h*x**2 + i*x

    # force start and end point to fit  https://stackoverflow.com/questions/33539287/how-to-force-specific-points-in-curve-fitting
    sigma = np.ones(len(pathX))
    sigma[[0, -1]] = 0.01

    # Fit the line equation
    # poptX, pcovX = curve_fit(quintic5deg, time, pathX, sigma=sigma)
    # poptY, pcovY = curve_fit(quintic5deg, time, pathY, sigma=sigma)
    # poptZ, pcovZ = curve_fit(quintic5deg, time, pathZ, sigma=sigma)
    # poptP, pcovP = curve_fit(quintic5deg, time, pathP, sigma=sigma)
    # poptQ, pcovQ = curve_fit(quintic5deg, time, pathQ, sigma=sigma)
    # poptR, pcovR = curve_fit(quintic5deg, time, pathR, sigma=sigma)
    # quintic5deg(timeSmooth[i], *poptX),
    timeSmooth = np.linspace(0, 1, 100)

    fig4, axes = plt.subplots(6, 1, sharex='all')
    axes[0].plot(time, pathX, 'ro')
    # axes[0].plot(time, quintic5deg(time, *poptX))
    axes[1].plot(time, pathY, 'ro')
    # axes[1].plot(time, quintic5deg(time, *poptY))
    axes[2].plot(time, pathZ, 'ro')
    # axes[2].plot(time, quintic5deg(time, *poptZ))
    axes[3].plot(time, pathP, 'ro')
    # axes[3].plot(time, quintic5deg(time, *poptP))
    axes[4].plot(time, pathQ, 'ro')
    # axes[4].plot(time, quintic5deg(time, *poptQ))
    axes[5].plot(time, pathR, 'ro')
    # axes[5].plot(time, quintic5deg(time, *poptR))
    plt.show()