""" 
Devplanner with RRT Informed with reject goal sampling
coppeliasim collision check is stupidly slow (0.8786161650000001 sec per check) but other code only take 0.017694391 sec what ?
so when copp is play mode is faster for KCD, ok.

"""
import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
import numpy as np
from planner_dev.copsim_rrt_component import Node, CopSimRRTComponent


class RRTInformedDev(CopSimRRTComponent):
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
        self.XSoln = []

    def planning(self):
        self.copHandle.start_sim()

        timePlanningStart = time.perf_counter_ns()
        itera = self.planner_rrt_informed()
        path = self.search_singledirection_path()
        timePlanningEnd = time.perf_counter_ns()

        # record performance
        self.perfMatrix["Planner Name"] = "Informed RRT Star"
        self.perfMatrix["Parameters"]["eta"] = self.eta 
        self.perfMatrix["Parameters"]["Max Iteration"] = self.maxIteration
        self.perfMatrix["Parameters"]["Rewire Radius"] = self.rewireRadius
        self.perfMatrix["Number of Node"] = len(self.treeVertex)
        self.perfMatrix["Total Planning Time"] = (timePlanningEnd-timePlanningStart) * 1e-9
        self.perfMatrix["Planning Time Only"] = self.perfMatrix["Total Planning Time"] - self.perfMatrix["KCD Time Spend"]* 1e-9
        self.perfMatrix["Average KCD Time"] = self.perfMatrix["KCD Time Spend"] / self.perfMatrix["Number of Collision Check"]

        self.copHandle.stop_sim()
        return path

    def planner_rrt_informed(self):
        for itera in range(self.maxIteration):
            print(itera)
            if len(self.XSoln) == 0:
                cBest = np.inf
                cBestPrevious = np.inf
            else:
                xSolnCost = [xSoln.parent.cost + self.cost_line(xSoln.parent, xSoln) + self.cost_line(xSoln, self.xApp) for xSoln in self.XSoln]
                print(f"==>> xSolnCost: \n{xSolnCost}")
                cBest = min(xSolnCost)
                if cBest < cBestPrevious : # this have nothing to do with planning itself, just for record performance data only
                    self.perfMatrix["Cost Graph"].append((itera, cBest))
                    cBestPrevious = cBest

            xRand = self.informed_sampling(self.xStart, self.xApp, cBest, biasToNode=None) #self.xApp) # this has bias careful
            xNearest = self.nearest_node(self.treeVertex, xRand)
            xNew = self.steer(xNearest, xRand, self.eta)
            xNew.parent = xNearest
            if self.is_config_in_region_of_config(xNew, self.xGoal, self.distGoalToApp) or \
                self.is_connect_config_in_region_of_config(xNew.parent, xNew, self.xGoal, self.distGoalToApp):
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

                # in approach region
                if self.is_config_in_region_of_config(xNew, self.xApp, radius=self.eta): # radius = 1
                    self.XSoln.append(xNew)

        return itera

    def search_singledirection_path(self):
        XNear = self.near(self.treeVertex, self.xApp, self.rewireRadius)
        for xNear in XNear:
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

    def informed_sampling(self, xStart, xGoal, cMax, biasToNode=None):
        if cMax < np.inf:
            cMin = self.distance_between_config(xStart, xGoal)
            print(cMax, cMin)
            xCenter = np.array([(xStart.x + xGoal.x) / 2,
                                (xStart.y + xGoal.y) / 2,
                                (xStart.z + xGoal.z) / 2,
                                (xStart.p + xGoal.p) / 2,
                                (xStart.q + xGoal.q) / 2,
                                (xStart.r + xGoal.r) / 2]).reshape(6, 1)

            L, C = self.rotation_to_world(xStart, xGoal, cMax, cMin)

            while True:
                xBall = self.unit_ball_sampling()
                xRand = (C@L@xBall) + xCenter
                xRand = Node(xRand[0, 0], xRand[1, 0], xRand[2, 0], xRand[3, 0], xRand[4, 0], xRand[5, 0])
                in_range = [(self.xMinRange < xRand.x < self.xMaxRange),
                            (self.yMinRange < xRand.y < self.yMaxRange),
                            (self.zMinRange < xRand.z < self.zMaxRange), 
                            (self.pMinRange < xRand.p < self.pMaxRange),
                            (self.qMinRange < xRand.q < self.qMaxRange),
                            (self.rMinRange < xRand.r < self.rMaxRange)]
                if all(in_range):
                    break
        else:
            if biasToNode is not None:
                xRand = self.bias_sampling(biasToNode)
            else:
                xRand = self.uni_sampling()
        return xRand

    def unit_ball_sampling(self):
        u = np.random.normal(0, 1, (1, 6 + 2))
        norm = np.linalg.norm(u, axis = -1, keepdims = True)
        u = u/norm
        return u[0,:6].reshape(6,1) #The first N coordinates are uniform in a unit N ball
    
    def rotation_to_world(self, xStart, xGoal, cMax, cMin):
        r1 = cMax / 2
        r2to6 = np.sqrt(cMax**2 - cMin**2) / 2
        L = np.diag([r1, r2to6, r2to6, r2to6, r2to6, r2to6])
        a1 = np.array([[(xGoal.x - xStart.x) / cMin],
                       [(xGoal.y - xStart.y) / cMin],
                       [(xGoal.z - xStart.z) / cMin],
                       [(xGoal.p - xStart.p) / cMin],
                       [(xGoal.q - xStart.q) / cMin],
                       [(xGoal.r - xStart.r) / cMin]])
        I1 = np.array([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
        M = a1 @ I1.T
        U, _, V_T = np.linalg.svd(M, True, True)
        C = U @ np.diag([1.0, 1.0, 1.0, 1.0, 1.0, np.linalg.det(U) * np.linalg.det(V_T.T)]) @ V_T
        return L, C

if __name__ == "__main__":
    np.random.seed(9)
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from planner_util.extract_path_class import extract_path_class_6d
    from target_localization.pre_record_value import thetaInit, thetaGoal2, thetaApp2, qAux, qGoal, wrap_to_pi
    from util.general_util import write_dict_to_file

    # Define pose
    thetaInit = thetaInit
    thetaGoal = wrap_to_pi(qGoal)
    thetaApp = wrap_to_pi(qAux)
    
    planner = RRTInformedDev(thetaInit, thetaApp, thetaGoal, eta=0.3, maxIteration=5000)
    path = planner.planning()
    write_dict_to_file(planner.perfMatrix, "./planner_dev/result_6d/result_6d_informedrrt.txt")
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

    # loop in simulation
    for i in range(len(pathX)):
        jointVal = np.array([pathX[i], pathY[i], pathZ[i], pathP[i], pathQ[i], pathR[i]]).reshape(6, 1)
        planner.copHandle.set_joint_value(jointVal)
        time.sleep(0.5)
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