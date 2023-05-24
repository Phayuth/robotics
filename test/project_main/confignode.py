import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from map.mapclass import map_val

class confignode:
    def __init__(self, q1, q2, pv) -> None:
        self.q1 = q1
        self.q2 = q2
        self.pv = pv

    def __repr__(self) -> str:
        return f"q1 = {self.q1} | q2 = {self.q2} | pv = {self.pv}"

class mapclass:
    def __init__(self) -> None:
        self.size = 100
        self.cell_size = 2*np.pi/self.size
        self.mapnode = [[confignode(map_val(i, 0, self.size, -np.pi, np.pi),map_val(j, 0, self.size, -np.pi, np.pi),np.random.uniform(low=0, high=1)) for i in range(self.size)] for j in range(self.size)]
    
    def map(self):
        return np.array(self.mapnode)
    
    def image(self):
        return np.array([[self.mapnode[i][j].pv for i in range(self.size)] for j in range(self.size)])
    
    def get_random_state(self):
        q1rand = np.random.randint(low=0, high=self.size) # random of discrete number in interger from uniform distribution
        q2rand = np.random.randint(low=0, high=self.size)
        return self.mapnode[q1rand][q2rand]

if __name__=="__main__":
    
    m = mapclass()
    plt.imshow(m.image())
    plt.show()

    randstate = m.get_random_state()
    print("==>> randstate: \n", randstate)
