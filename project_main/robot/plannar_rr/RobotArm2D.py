import numpy as np

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