import numpy as np
import matplotlib.pyplot as plt

class Robot(object):

    def __init__(self, base_position = None, link_lenths = None):

        self.base_position = base_position
        self.link_lenths = np.array(link_lenths)

    def robot_position(self, theta1, theta2):

        position = []
        position.append(self.base_position)

        theta1 = (theta1 * np.pi) / 180
        theta2 = (theta2 * np.pi) / 180

        x1 = self.base_position[0] + self.link_lenths[0] * np.cos(theta1)
        y1 = self.base_position[1] + self.link_lenths[0] * np.sin(theta1)

        position.append([x1, y1])

        x2 = self.base_position[0] + self.link_lenths[0] * np.cos(theta1) + self.link_lenths[1] * np.cos(theta1 + theta2)
        y2 = self.base_position[1] + self.link_lenths[0] * np.sin(theta1) + self.link_lenths[1] * np.sin(theta1 + theta2)

        position.append([x2, y2])

        return position
    
if __name__=="__main__":
    base_position = [15, 15]
    link_lenths = [5, 5]
    robot = Robot(base_position, link_lenths)

    plt.figure(figsize=(10,10))
    plt.axes().set_aspect('equal')
    r1 = robot.robot_position(90,0)
    plt.plot([robot.base_position[0], r1[0][0]],[robot.base_position[1], r1[0][1]] , "b", linewidth=8)
    plt.plot([r1[0][0], r1[1][0]],[r1[0][1], r1[1][1]] , "b", linewidth=8)
    plt.plot([r1[1][0], r1[2][0]],[r1[1][1], r1[2][1]] , "r", linewidth=8)
    plt.gca().invert_yaxis()
    plt.show()