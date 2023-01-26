import matplotlib.pyplot as plt
import numpy as np

class path():
    def __init__(self,start_node,end_node):
        self.start = start_node
        self.end   = end_node
        self.data = [[1,1,1,1,1,1,1,1,1,1],
                     [1,0,0,0,0,0,0,0,0,1],
                     [1,0,0,0,0,0,0,0,0,1],
                     [1,0,0,0,0,0,0,0,0,1],
                     [1,0,0,0,0,0,0,0,0,1],
                     [1,0,0,0,0,0,0,0,0,1],
                     [1,0,0,0,0,0,0,0,0,1],
                     [1,0,0,0,0,0,0,0,0,1],
                     [1,0,0,0,0,0,0,0,0,1],
                     [1,1,1,1,1,1,1,1,1,1]]
        self.maze = np.array(self.data)
    
    def start_end(self):
        self.maze[self.start[0],self.start[1]] = 2
        self.maze[self.end[0],self.end[1]] = 2


    def to_path(self,x_coord,y_coord):
        self.maze[x_coord,y_coord] = 2


    def plot(self):
        plt.imshow(self.maze)
        plt.show()



pp = path([3,3],[6,6])
pp.start_end()
pp.to_path(5,5)
pp.plot()