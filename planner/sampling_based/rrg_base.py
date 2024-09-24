import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

from planner.sampling_based.prm_component import PRMComponent, PRMPlotter


class RRGBase(PRMComponent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_graph(self, numNodes):  # build by adding more node to existing roadmap (RRG)
        while True:
            xRand = self.uni_sampling()
            if not self.is_config_in_collision(xRand):
                self.nodes.append(xRand)
                break

        for _ in range(numNodes):
            xRand = self.uni_sampling()
            xNearest = self.nearest_node(self.nodes, xRand)
            xNew = self.steer(xNearest, xRand, self.eta)
            XNear, _ = self.near(self.nodes, xNew)
            for xNear in XNear:
                if not self.is_collision(xNew, xNear):
                    cost = self.distance_between_config(xNew, xNear)
                    if not xNear in xNew.edgeNodes:
                        xNew.edgeNodes.append(xNear)
                        xNew.edgeCosts.append(cost)
                    if not xNew in xNear.edgeNodes:
                        xNear.edgeNodes.append(xNew)
                        xNear.edgeCosts.append(cost)
                    if not xNew in self.nodes:
                        self.nodes.append(xNew)
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
    from simulator.sim_rectangle import TaskSpace2DSimulator

    robotsim = TaskSpace2DSimulator()
    prm = RRGBase(simulator=robotsim, eta=0.3, subEta=0.05)
    prm.build_graph(1000)

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
