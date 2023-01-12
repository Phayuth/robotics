import numpy as np

class node:
    def __init__(self,x,y,child,parent):
        self.x = x
        self.y = y
        self.child = child
        self.parent = parent


node1 = node(0,0,[[0,1],[1,0],[1,1]],[])
node2 = node(0,1,[[1,1]],[[0,0]])
node3 = node(1,0,[[0,0]],[[0,0]])
node4 = node(1,1,[[2,2]],[[0,0],[1,0],[0,1]])
node5 = node(2,2,[[]],[[1,1]])

graph = [node1,node2,node3,node4,node5]