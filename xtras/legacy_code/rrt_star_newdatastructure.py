import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
from planner.sampling_based.rrt_component import Node, RRTComponent


class RRTStar(RRTComponent):

    def __init__(self, xStart, xApp, xGoal, config):
        super().__init__(config)
        # start, aux, goal node
        self.xStart = Node(xStart)
        self.xGoal = Node(xGoal)
        self.xApp = Node(xApp)

        # planner properties
        self.treeDataStructure = np.empty((self.configDoF, self.maxIteration))
        self.treeDataStructure[:, 0, np.newaxis] = self.xStart.config
        self.treeVertex = [self.xStart]
        self.distGoalToApp = self.distance_between_config(self.xGoal, self.xApp)

        # local sampling properties
        self.localOptEnable = config["localOptEnable"]
        if self.localOptEnable:
            self.anchorPath = None
            self.localPath = None
            self.numSegSamplingNode = None

        # solutions
        self.XInGoalRegion = []

    # @RRTComponent.catch_key_interrupt
    # def start(self):
    #     for itera in range(self.maxIteration):
    #         self.cBestNow = self.cbest_single_tree(self.XInGoalRegion, self.xApp, itera)  # save cost graph

    #         if self.localOptEnable:
    #             if len(self.XInGoalRegion) == 0:
    #                 xRand = self.bias_uniform_sampling(self.xApp, len(self.XInGoalRegion))
    #             else:
    #                 xRand = self.local_path_sampling(self.anchorPath, self.localPath, self.numSegSamplingNode)
    #         else:
    #             xRand = self.bias_uniform_sampling(self.xApp, len(self.XInGoalRegion))

    #         xNearest, vertexDistList = self.nearest_node(self.treeVertex, xRand, returnDistList=True)
    #         xNew, xNewIsxRand = self.steer(xNearest, xRand, self.eta, returnIsReached=True)
    #         if self.is_collision_and_in_goal_region(xNearest, xNew, self.xGoal, self.distGoalToApp):
    #             continue
    #         xNew.parent = xNearest
    #         xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
    #         xNearest.child.append(xNew)

    #         # rrtstar
    #         self.star_optimizer(self.treeVertex, xNew, self.rewireRadius, xNewIsxRand, vertexDistList)

    #         # in goal region
    #         if self.is_config_in_region_of_config(xNew, self.xApp, radius=self.nearGoalRadius):
    #             self.XInGoalRegion.append(xNew)
    #             if self.localOptEnable:
    #                 if self.anchorPath is None:
    #                     self.localPath = self.search_backtrack_single_directional_path(self.XInGoalRegion[0], self.xApp)
    #                     self.numSegSamplingNode = len(self.localPath) - 1
    #                     self.anchorPath = self.segment_interpolation_between_config(self.xStart, self.xApp, self.numSegSamplingNode, includexStart=True)

    #         if self.termination_check(self.XInGoalRegion):
    #             break

    # def get_path(self):
    #     return self.search_best_cost_singledirection_path(backFromNode=self.xApp, treeVertexList=self.XInGoalRegion, attachNode=self.xGoal)

    def nearest_node_new(self, xCheck):
        c = xCheck.config
        print(f"> c: {c}")
        v = self.treeDataStructure[:, 0 : len(self.treeVertex)]
        print(f"> v: {v}")
        dist = np.linalg.norm(v - c)
        print(f"> dist: {dist}")

    @RRTComponent.catch_key_interrupt
    def start(self):
        for itera in range(self.maxIteration):

            xRand = self.uni_sampling()

            # add tree to datastructure
            self.treeVertex.append(xRand)
            dd = self.treeDataStructure[:, itera, np.newaxis]
            print(itera+1)
            # self.treeDataStructure[:, len(self.treeVertex)-1, np.newaxis] = xRand.config

        # self.nearest_node_new(xRand)

if __name__ == "__main__":
    np.random.seed(9)
    from spatial_geometry.spatial_transformation import RigidBodyTransformation as rbt
    from simulator.sim_planar_rr import RobotArm2DSimulator
    from planner.sampling_based.rrt_plannerapi import RRTPlannerAPI
    from spatial_geometry.utils import Utils

    sim = RobotArm2DSimulator()

    planconfig = {
        "eta": 0.3,
        "subEta": 0.05,
        "maxIteration": 50,
        "simulator": sim,
        "nearGoalRadius": None,
        "rewireRadius": None,
        "endIterationID": 1,
        "printDebug": True,
        "localOptEnable": True,
    }

    xStart = np.array([0.2, 0.2]).reshape(2, 1)
    xGoal = np.array([np.pi / 2, 0]).reshape(2, 1)
    xApp = np.array([np.pi / 2 - 0.1, 0.2]).reshape(2, 1)

    rrtstar = RRTStar(xStart, xGoal, xApp, planconfig)
    rrtstar.start()
