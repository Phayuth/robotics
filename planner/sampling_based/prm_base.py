import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

from planner.sampling_based.prm_component import PRMComponent


class PRMBase(PRMComponent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # nodes, roadmap
        self.roadmap = []
        self.nodes = []

    def build_roadmap_batch(self, numNodes):
        # build from stratch (batch PRM)
        XRand = [self.uni_sampling() for _ in range(numNodes)]
        for xRand in XRand:
            XNear, distXNear = self.near(XRand, xRand, None, None)
            for xNear in XNear:
                if not self.is_collision(xRand, xNear):
                    cost = self.distance_between_config(xRand, xNear)
                    if not xNear in xRand.edgeNodes:
                        xRand.edgeNodes.append(xNear)
                        xRand.edgeCosts.append(cost)
                    if not xRand in xNear.edgeNodes:
                        xNear.edgeNodes.append(xRand)
                        xNear.edgeCosts.append(cost)
                    if not xRand in self.nodes:
                        self.nodes.append(xRand)
                    if not xNear in self.nodes:
                        self.nodes.append(xNear)

        print("Building Roadmap Done.")
        print(f"Current NumNodes = [{len(self.nodes)}]")

    def build_roadmap_rdg(self, numNodes):
        # build by adding more node to existing roadmap (RDG)
        for _ in range(numNodes):
            xRand = self.uni_sampling()
            XNear, distXNear = self.near(self.nodes, xRand, None, None)
            for xNear in XNear:
                if not self.is_collision(xRand, xNear):
                    cost = self.distance_between_config(xRand, xNear)
                    if not xNear in xRand.edgeNodes:
                        xRand.edgeNodes.append(xNear)
                        xRand.edgeCosts.append(cost)
                    if not xRand in xNear.edgeNodes:
                        xNear.edgeNodes.append(xRand)
                        xNear.edgeCosts.append(cost)
                    if not xRand in self.nodes:
                        self.nodes.append(xRand)
                    if not xNear in self.nodes:
                        self.nodes.append(xNear)

        print("Building Roadmap Done.")
        print(f"Current NumNodes = [{len(self.nodes)}]")


if __name__ == "__main__":
    import os
    import sys

    wd = os.path.abspath(os.getcwd())
    sys.path.append(str(wd))
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(9)
    from planner.sampling_based.prm_plotter import PRMPlotter
    from simulator.sim_rectangle import TaskSpace2DSimulator

    robotsim = TaskSpace2DSimulator()
    prm = PRMBase(simulator=robotsim, eta=0.3, subEta=0.05)
    prm.build_roadmap_batch(1000)

    xStart = np.array([0.0, 0.0]).reshape(2, 1)
    xGoal = np.array([-2.3, 2.3]).reshape(2, 1)
    path = prm.query(xStart, xGoal, prm.nodes, searcher="ast")
    print(f"> path: {path}")

    fig, ax = plt.subplots()
    ax.set_axis_off()
    fig.set_size_inches(w=3.40067, h=3.40067)
    fig.tight_layout()
    plt.xlim((-np.pi, np.pi))
    plt.ylim((-np.pi, np.pi))
    PRMPlotter.plot_2d_config(path, prm, ax)
    plt.show()
