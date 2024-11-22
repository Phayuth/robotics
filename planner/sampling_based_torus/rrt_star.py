import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
from planner.sampling_based_torus.rrt_component import Node, RRTTorusRedundantComponent


class RRTTorusRedundantStar(RRTTorusRedundantComponent):

    def __init__(self, xStart, xApp, xGoal, config):
        super().__init__(config)
        # start, aux, goal node
        self.xStart = Node(xStart)
        self.xGoal = Node(xGoal)
        self.xApp = Node(xApp)

        # planner properties
        self.treeVertex = [self.xStart]

        # solutions
        self.XInGoalRegion = []

    @RRTTorusRedundantComponent.catch_key_interrupt
    def start(self):
        for itera in range(self.maxIteration):
            self.cBestNow = self.cbest_single_tree(self.XInGoalRegion, self.xApp, itera)  # save cost graph

            xRand = self.uni_sampling()
            xNearest = self.nearest_wrap_node(self.treeVertex, xRand)
            xNew = self.steer_wrap(xNearest, xRand, self.eta)
            if self.is_collision(xNearest, xNew):
                continue
            xNew.parent = xNearest
            xNew.cost = xNew.parent.cost + self.cost_line(xNew, xNew.parent)
            xNearest.child.append(xNew)

            # rrtstar
            self.star_optimizer(self.treeVertex, xNew, self.rewireRadius)

            # in goal region
            if self.is_config_in_region_of_config(xNew, self.xApp, radius=self.nearGoalRadius):
                self.XInGoalRegion.append(xNew)

            if self.termination_check(self.XInGoalRegion):
                break

    def get_path(self):
        return self.search_best_cost_singledirection_path(backFromNode=self.xApp, treeVertexList=self.XInGoalRegion, attachNode=self.xGoal)

    def get_path_array(self, pathNodesSeq): # this is wrong becuase it doesn't unwrap. but i want to demo it.
        pathTemp = [node.config for node in pathNodesSeq]
        return np.hstack(pathTemp)  # shape=(numDoF, numSeq)


if __name__=="__main__":
    np.random.seed(9)
    import matplotlib.pyplot as plt
    from simulator.sim_planar_rr import RobotArm2DSimulator
    from planner.sampling_based_torus.rrt_component import RRTPlotter

    sim = RobotArm2DSimulator()

    plannarConfigDualTreea = {
        "planner": "does not matter",
        "eta": 0.3,
        "subEta": 0.05,
        "maxIteration": 2000,
        "simulator": sim,
        "nearGoalRadius": None,
        "rewireRadius": None,
        "endIterationID": 1,
        "printDebug": True,
        "localOptEnable": False
    }

    xStart = np.array([0.0, 2.8]).reshape(2, 1)
    xApp = np.array([-1.2, -2.3]).reshape(2, 1)
    xGoal = np.array([-1.2, -2.3]).reshape(2, 1)

    pm = RRTTorusRedundantStar(xStart, xApp, xGoal, plannarConfigDualTreea)
    path = pm.begin_planner()

    fig, ax = plt.subplots()
    s = 2.5
    fig.set_size_inches(w=s * 3.40067, h=s * 3.40067)
    fig.tight_layout()
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)
    RRTPlotter.plot_2d_complete(pm, path, ax)
    plt.show()


    patharray = pm.get_path_array(path)
    time = np.linspace(0, 1, num=patharray.shape[1])
    print(f"> time.shape: {time.shape}")
    print(patharray.shape)

    fig, axs = plt.subplots(patharray.shape[0], 1, figsize=(10, 15), sharex=True)
    for i in range(patharray.shape[0]):
        axs[i].plot(time, patharray[i, :], color="blue", marker="o", linestyle="dashed", linewidth=2, markersize=12, label=f"Joint Pos {i+1}")
        axs[i].set_ylabel(f"JPos {i+1}")
        axs[i].set_xlim(time[0], time[-1])
        axs[i].legend(loc="upper right")
        axs[i].grid(True)
    axs[-1].set_xlabel("Time")
    fig.suptitle("Joint Position")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
