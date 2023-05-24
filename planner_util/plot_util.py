import matplotlib.pyplot as plt


def plot_tree(treeVertex, path):
    for vertex in treeVertex:
        plt.scatter(vertex.x, vertex.y, color="sandybrown")
        if vertex.parent == None:
            pass
        else:
            plt.plot([vertex.x, vertex.parent.x], [vertex.y, vertex.parent.y], color="green")

    if path:
        plt.plot([node.x for node in path], [node.y for node in path], color='blue')


def plot_tree_3d(treeVertex, path):
    ax = plt.axes(projection='3d')

    for vertex in treeVertex:
        # ax.scatter(vertex.x, vertex.y, vertex.z, color="sandybrown")
        if vertex.parent == None:
            pass
        else:
            ax.plot3D([vertex.x, vertex.parent.x], [vertex.y, vertex.parent.y], [vertex.z, vertex.parent.z], "green")
   
    if path:
        ax.plot3D([node.x for node in path], [node.y for node in path], [node.x for node in path], color='blue')
