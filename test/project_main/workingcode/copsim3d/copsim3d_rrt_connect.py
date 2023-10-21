""" 
Path Planning Development for UR5 and UR5e robot with coppeliasim

"""
import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
import numpy as np
from planner_dev.rrt_component import Node, RRTComponent


class RRTConnectDev(RRTComponent):
    def __init__(self, xStartFull, xAppFull, xGoalFull, eta=0.3, maxIteration=1000) -> None:
        super().__init__()
        # start, aux, goal node
        self.xStart = Node(xStartFull[0, 0], xStartFull[1, 0], xStartFull[2, 0])
        self.xGoal = Node(xGoalFull[0, 0], xGoalFull[1, 0], xGoalFull[2, 0])
        self.xApp = Node(xAppFull[0, 0], xAppFull[1, 0], xAppFull[2, 0])

        self.joint4Fixed = xAppFull[3, 0]
        self.joint5Fixed = xAppFull[4, 0]
        self.joint6Fixed = xAppFull[5, 0]

        self.eta = eta
        self.subEta = 0.05
        self.maxIteration = maxIteration
        self.treeVertexStart = [self.xStart]
        self.treeVertexGoal = [self.xApp]
        self.treeSwapFlag = True
        self.connectNodeStart = None
        self.connectNodeGoal = None
        self.distGoalToApp = self.distance_between_config(self.xGoal, self.xApp)

    def planning(self):
        self.copHandle.start_sim()
        timePlanningStart = time.perf_counter_ns()
        itera = self.planner_rrt_connect_app()
        path = self.search_backtrack_single_directional_path(backFromNode=self.xApp, attachNode=self.xGoal)
        timePlanningEnd = time.perf_counter_ns()

        # record performance
        self.perfMatrix["Planner Name"] = "RRT-Connect NewMod"
        self.perfMatrix["Parameters"]["eta"] = self.eta 
        self.perfMatrix["Parameters"]["subEta"] = self.subEta
        self.perfMatrix["Parameters"]["Max Iteration"] = self.maxIteration
        self.perfMatrix["Parameters"]["Rewire Radius"] = 0.0
        self.perfMatrix["Number of Node"] = len(self.treeVertexStart) + len(self.treeVertexGoal)
        self.perfMatrix["Total Planning Time"] = (timePlanningEnd-timePlanningStart) * 1e-9
        self.perfMatrix["KCD Time Spend"] = self.perfMatrix["KCD Time Spend"]* 1e-9
        self.perfMatrix["Planning Time Only"] = self.perfMatrix["Total Planning Time"] - self.perfMatrix["KCD Time Spend"]
        self.perfMatrix["Average KCD Time"] = self.perfMatrix["KCD Time Spend"] / self.perfMatrix["Number of Collision Check"]

        self.copHandle.stop_sim()
        return path

    def planner_rrt_connect_app(self):
        for itera in range(self.maxIteration):
            print(itera)
            if self.treeSwapFlag is True: # Init tree side
                xRand = self.uni_sampling()
                xNearest = self.nearest_node(self.treeVertexStart, xRand)
                xNew = self.steer(xNearest, xRand, self.eta)

                if not self.is_collision_and_in_goal_region(xNearest, xNew, self.xGoal, self.distGoalToApp):
                    xNew.parent = xNearest
                    xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
                    xNearest.child.append(xNew)
                    self.treeVertexStart.append(xNew)
                    xNearestPrime = self.nearest_node(self.treeVertexGoal, xNew)
                    xNewPrime = self.steer(xNearestPrime, xNew, self.eta)

                    if not self.is_collision_and_in_goal_region(xNearestPrime, xNewPrime, self.xGoal, self.distGoalToApp):
                        xNewPrime.parent = xNearestPrime
                        xNewPrime.cost = xNewPrime.parent.cost + self.cost_line(xNewPrime, xNewPrime.parent)
                        xNearestPrime.child.append(xNewPrime)
                        self.treeVertexGoal.append(xNewPrime)

                        while True:
                            xNewPPrime = self.steer(xNewPrime, xNew, self.eta)

                            if self.is_collision_and_in_goal_region(xNewPrime, xNewPPrime, self.xGoal, self.distGoalToApp):
                                break

                            # if the 2 node meet, then break
                            if self.distance_between_config(xNewPPrime, xNew) < 1e-3:
                                self.connectNodeGoal = xNewPrime
                                self.connectNodeStart = xNew
                                break

                            # if not collision then free to add
                            else:
                                xNewPPrime.parent = xNewPrime
                                xNewPPrime.cost = xNewPPrime.parent.cost + self.cost_line(xNewPPrime, xNewPPrime.parent)
                                xNewPrime.child.append(xNewPPrime)
                                self.treeVertexGoal.append(xNewPPrime)
                                xNewPrime = xNewPPrime

                if self.connectNodeGoal is not None and self.connectNodeStart is not None:
                    self.reparent_merge_tree()
                    break

                self.tree_swap_flag()

            elif self.treeSwapFlag is False: # App tree side
                xRand = self.uni_sampling()
                xNearest = self.nearest_node(self.treeVertexGoal, xRand)
                xNew = self.steer(xNearest, xRand, self.eta)

                if not self.is_collision_and_in_goal_region(xNearest, xNew, self.xGoal, self.distGoalToApp):
                    xNew.parent = xNearest
                    xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
                    xNearest.child.append(xNew)
                    self.treeVertexGoal.append(xNew)
                    xNearestPrime = self.nearest_node(self.treeVertexStart, xNew)
                    xNewPrime = self.steer(xNearestPrime, xNew, self.eta)

                    if not self.is_collision_and_in_goal_region(xNearestPrime, xNewPrime, self.xGoal, self.distGoalToApp):
                        xNewPrime.parent = xNearestPrime
                        xNewPrime.cost = xNewPrime.parent.cost + self.cost_line(xNewPrime, xNewPrime.parent)
                        xNearestPrime.child.append(xNewPrime)
                        self.treeVertexStart.append(xNewPrime)

                        while True:
                            xNewPPrime = self.steer(xNewPrime, xNew, self.eta)

                            if self.is_collision_and_in_goal_region(xNewPrime, xNewPPrime, self.xGoal, self.distGoalToApp):
                                break

                            # if the 2 node meet, then break
                            if self.distance_between_config(xNewPPrime, xNew) < 1e-3:
                                self.connectNodeGoal = xNew
                                self.connectNodeStart = xNewPrime
                                break

                            # if not collision then free to add
                            else:
                                xNewPPrime.parent = xNewPrime
                                xNewPPrime.cost = xNewPPrime.parent.cost + self.cost_line(xNewPPrime, xNewPPrime.parent)
                                xNewPrime.child.append(xNewPPrime)
                                self.treeVertexStart.append(xNewPPrime)
                                xNewPrime = xNewPPrime

                if self.connectNodeGoal is not None and self.connectNodeStart is not None:
                    self.reparent_merge_tree()
                    break

                self.tree_swap_flag()

        return itera

    def reparent_merge_tree(self):
        xTobeParent = self.connectNodeStart
        xNow = self.connectNodeGoal

        while True:
            if xNow.parent is None:
                xNow.parent = xTobeParent
                xNow.cost = xNow.parent.cost + self.cost_line(xNow.parent, xNow)
                xTobeParent.child.append(xNow)
                self.update_child_recursion(xNow) #update child 
                # xParentSave.child.remove(xNow) # Xnow is Xapp which has no parent so we dont have to do this
                self.treeVertexStart.append(xNow)
                break

            xParentSave = xNow.parent
            xNow.parent = xTobeParent
            xNow.cost = xNow.parent.cost + self.cost_line(xNow.parent, xNow)
            xTobeParent.child.append(xNow)
            self.update_child_recursion(xNow) #update child 
            xParentSave.child.remove(xNow)
            self.treeVertexStart.append(xNow)

            # update for next iteration
            xTobeParent = xNow
            xNow = xParentSave

    def update_child_recursion(self, node):
        for child_node in node.child:
            child_node.cost = child_node.parent.cost + self.cost_line(child_node.parent, child_node)
            self.treeVertexStart.append(child_node)
            self.update_child_recursion(child_node)

    def plot_tree(self, path, ax):
        self.plot_single_tree(self.treeVertexStart, path, ax, obstacle_plot=False)

if __name__ == "__main__":
    np.random.seed(9)
    import matplotlib.pyplot as plt
    from planner.extract_path_class import extract_path_class_3d
    from datasave.joint_value.pre_record_value import wrap_to_pi, newThetaInit, newThetaApp, newThetaGoal
    from copsim.arm_api import UR5eStateArmCoppeliaSimAPI
    from util.general_util import write_dict_to_file

    # Define pose
    thetaInit = newThetaInit
    thetaGoal = wrap_to_pi(newThetaGoal)
    thetaApp = wrap_to_pi(newThetaApp)
    
    planner = RRTConnectDev(thetaInit, thetaApp, thetaGoal, eta=0.2, maxIteration=5000)
    path = planner.planning()
    print(planner.perfMatrix)
    # write_dict_to_file(planner.perfMatrix, "./planner_dev/result_6d/result_6d_rrtstar.txt")
    print(f"==>> path: \n{path}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    planner.plot_tree(path, ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
    # time.sleep(3)

    # # play back
    planner.copHandle.start_sim()
    pathX, pathY, pathZ = extract_path_class_3d(path)
    pathX = np.array(pathX)
    pathY = np.array(pathY)
    pathZ = np.array(pathZ)

    armState = UR5eStateArmCoppeliaSimAPI()
    armState.set_goal_joint_value(thetaGoal)
    armState.set_aux_joint_value(thetaApp)
    armState.set_start_joint_value(thetaInit)
    # time.sleep(100)
    for i in range(len(pathX)):
        jointVal = np.array([pathX[i], pathY[i], pathZ[i], planner.joint4Fixed, planner.joint5Fixed, planner.joint6Fixed]).reshape(6, 1)
        planner.copHandle.set_joint_value(jointVal)
        time.sleep(0.1)
        # triggers next simulation step
        # client.step()

    # stop simulation
    planner.copHandle.stop_sim()