import matplotlib.pyplot as plt


def plot_tree(treeVertex, xStart, xGoal):
    # plot tree vertex and branches
    for j in treeVertex:
        plt.scatter(j.x, j.y, color="red")  # vertex
        if j.parent == None:
            j.parent = xStart
        plt.plot([j.x, j.parent.x], [j.y, j.parent.y], color="green")  # branch

    # plot start and goal Node
    plt.scatter([xStart.x, xGoal.x], [xStart.y, xGoal.y], color='cyan')