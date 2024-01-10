"""
Path Pure Pursuit Control for Differential Drive Mobile Robot
Reference : https://wiki.purduesigbots.com/software/control-algorithms/basic-pure-pursuit

"""
import numpy as np


class DifferentialDrivePurePursuitController:

    def __init__(self, path, loopMode=False) -> None:
        # path
        self.path = path
        self.loopMode = loopMode  # true if we want to loop path over and over

        # controller params
        self.lastFoundIndex = 0
        self.lookAheadDis = 0.8
        self.startingIndex = 0

        # controller tuning
        self.linearVel = 0.3  # 100
        self.Kp = 0.5  # orientation control 5

    def distance_point_to_point(self, pt1, pt2):
        distance = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        return distance

    def kinematic_control(self, currentPos):
        if self.loopMode:  # for the animation to loop
            if self.lastFoundIndex >= len(self.path) - 2:
                self.lastFoundIndex = 0

        # extract currentX and currentY
        currentX = currentPos[0, 0]
        currentY = currentPos[1, 0]

        # use for loop to search intersections
        intersectFound = False
        self.startingIndex = self.lastFoundIndex

        for i in range(self.startingIndex, len(self.path) - 1):

            # beginning of line-circle intersection code
            x1 = self.path[i][0] - currentX
            y1 = self.path[i][1] - currentY
            x2 = self.path[i + 1][0] - currentX
            y2 = self.path[i + 1][1] - currentY
            dx = x2 - x1
            dy = y2 - y1
            dr = np.linalg.norm([dx, dy])
            D = x1 * y2 - x2 * y1
            discriminant = (self.lookAheadDis**2) * (dr**2) - D**2

            if discriminant >= 0:
                sol_x1 = (D * dy + np.sign(dy) * dx * np.sqrt(discriminant)) / dr**2
                sol_x2 = (D * dy - np.sign(dy) * dx * np.sqrt(discriminant)) / dr**2
                sol_y1 = (-D * dx + abs(dy) * np.sqrt(discriminant)) / dr**2
                sol_y2 = (-D * dx - abs(dy) * np.sqrt(discriminant)) / dr**2

                sol_pt1 = [sol_x1 + currentX, sol_y1 + currentY]
                sol_pt2 = [sol_x2 + currentX, sol_y2 + currentY]
                # end of line-circle intersection code

                minX = min(self.path[i][0], self.path[i + 1][0])
                minY = min(self.path[i][1], self.path[i + 1][1])
                maxX = max(self.path[i][0], self.path[i + 1][0])
                maxY = max(self.path[i][1], self.path[i + 1][1])

                # if one or both of the solutions are in range
                if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) or ((minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)):

                    foundIntersection = True

                    # if both solutions are in range, check which one is better
                    if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) and ((minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)):
                        # make the decision by compare the distance between the intersections and the next point in path
                        if self.distance_point_to_point(sol_pt1, self.path[i + 1]) < self.distance_point_to_point(sol_pt2, self.path[i + 1]):
                            goalPt = sol_pt1
                        else:
                            goalPt = sol_pt2

                    # if not both solutions are in range, take the one that's in range
                    else:
                        # if solution pt1 is in range, set that as goal point
                        if (minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY):
                            goalPt = sol_pt1
                        else:
                            goalPt = sol_pt2

                    # only exit loop if the solution pt found is closer to the next pt in path than the current pos
                    if self.distance_point_to_point(goalPt, self.path[i + 1]) < self.distance_point_to_point([currentX, currentY], self.path[i + 1]):
                        # update lastFoundIndex and exit
                        self.lastFoundIndex = i
                        break
                    else:
                        # in case for some reason the robot cannot find intersection in the next path segment, but we also don't want it to go backward
                        self.lastFoundIndex = i + 1

                # if no solutions are in range
                else:
                    foundIntersection = False
                    # no new intersection found, potentially deviated from the path
                    # follow path[lastFoundIndex]
                    goalPt = [self.path[self.lastFoundIndex][0], self.path[self.lastFoundIndex][1]]

            # if determinant < 0
            else:
                foundIntersection = False
                # no new intersection found, potentially deviated from the path
                # follow path[lastFoundIndex]
                goalPt = [self.path[self.lastFoundIndex][0], self.path[self.lastFoundIndex][1]]

        # obtained goal point, now compute turn vel
        # calculate absTargetAngle with the atan2 function
        absTargetAngle = np.arctan2(goalPt[1] - currentPos[1, 0], goalPt[0] - currentPos[0, 0])
        if absTargetAngle < 0:
            absTargetAngle += 2 * np.pi

        # compute turn error by finding the minimum angle
        turnError = absTargetAngle - currentPos[2, 0]
        if turnError > np.pi or turnError < -np.pi:
            turnError = -1 * np.sign(turnError) * (2 * np.pi - abs(turnError))

        # controller
        turnVel = self.Kp * turnError

        return np.array([[self.linearVel], [turnVel]])