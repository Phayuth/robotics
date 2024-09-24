import os
import sys
sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
import matplotlib.pyplot as plt
from spatial_geometry.spatial_transformation import RigidBodyTransformation as rbt


class ShapeAABB:

    def __init__(self, xMin, yMin, zMin, xMax, yMax, zMax):
        self.xMin = xMin
        self.yMin = yMin
        self.zMin = zMin
        self.xMax = xMax
        self.yMax = yMax
        self.zMax = zMax


class ShapeRectangle:

    def __init__(self, x, y, h, w, angle=0.0, p=None):
        self.x = x
        self.y = y
        self.h = h
        self.w = w
        self.angle = angle
        self.p = p  # probability value

    def plot(self):
        if self.p == 1:
            color = "r"
        else:
            color = "darkcyan"

        # v1 = np.array([0, 0]).reshape(2, 1)
        # v2 = np.array([self.w, 0]).reshape(2, 1)
        # v3 = np.array([self.w, self.h]).reshape(2, 1)
        # v4 = np.array([0, self.h]).reshape(2, 1)

        # v1 = rbt.rot2d(self.angle) @ v1 + np.array([self.x, self.y]).reshape(2, 1)
        # v2 = rbt.rot2d(self.angle) @ v2 + np.array([self.x, self.y]).reshape(2, 1)
        # v3 = rbt.rot2d(self.angle) @ v3 + np.array([self.x, self.y]).reshape(2, 1)
        # v4 = rbt.rot2d(self.angle) @ v4 + np.array([self.x, self.y]).reshape(2, 1)

        # plt.plot([v1[0, 0], v2[0, 0]], [v1[1, 0], v2[1, 0]], c=color)
        # plt.plot([v2[0, 0], v3[0, 0]], [v2[1, 0], v3[1, 0]], c=color)
        # plt.plot([v3[0, 0], v4[0, 0]], [v3[1, 0], v4[1, 0]], c=color)
        # plt.plot([v4[0, 0], v1[0, 0]], [v4[1, 0], v1[1, 0]], c=color)

        # rec = plt.Rectangle((self.x, self.y), self.w, self.h, angle=self.angle, edgecolor=color, fill=False, hatch="x")
        rec = plt.Rectangle((self.x, self.y), self.w, self.h, angle=self.angle, edgecolor=color, facecolor=color, fill=True)
        plt.gca().add_patch(rec)


class ShapePoint3D:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class ShapePoint2D:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def plot(self):
        plt.scatter(self.x, self.y)


class ShapeLine2D:

    def __init__(self, xs, ys, xe, ye):
        self.xs = xs
        self.ys = ys
        self.xe = xe
        self.ye = ye

    def plot(self):
        plt.plot([self.xs, self.xe], [self.ys, self.ye])


class ShapeSphere:

    def __init__(self, x, y, z, r):
        self.x = x
        self.y = y
        self.z = z
        self.r = r


class ShapeCircle:

    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

    def plot(self):
        thetaCoord = np.linspace(0, 2 * np.pi, 90)
        xCoord, yCoord = rbt.conv_polar_to_cartesian(self.r, thetaCoord, self.x, self.y)
        plt.plot(xCoord, yCoord, "g--")


class ShapeEllipse:

    def __init__(self, x, y, a, b, rotMat) -> None:
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        self.rotMat = rotMat

    def plot(self):
        t = np.linspace(0, 2 * np.pi, 100)
        x_local = self.a * np.cos(t)
        y_local = self.b * np.sin(t)
        local_points = np.vstack((x_local, y_local))
        rotated_points = self.rotMat @ local_points
        translated_points = rotated_points + np.array([[self.x], [self.y]])
        plt.plot(translated_points[0], translated_points[1], "g--")


class ShapeTriangle:

    def __init__(self, vertixA, vertixB, vertixC):
        self.vertixAX = vertixA[0]
        self.vertixAY = vertixA[1]
        self.vertixBX = vertixB[0]
        self.vertixBY = vertixB[1]
        self.vertixCX = vertixC[0]
        self.vertixCY = vertixC[1]


class ShapeSuperEllipsoid2D:

    def __init__(self, a, b, n) -> None:
        self.a = a
        self.b = b
        self.n = n

    def plot(self):
        theta = np.linspace(0, 2 * np.pi, 360)
        x = ((abs(np.cos(theta)))**(2 / self.n)) * (self.a * np.sign(np.cos(theta)))
        y = ((abs(np.sin(theta)))**(2 / self.n)) * (self.b * np.sign(np.sin(theta)))
        plt.plot(x, y, "g--")


class ShapeCollision:

    def intersect_aabb_v_aabb(a, b):
        return (a.xMin <= b.xMax and a.xMax >= b.xMin and a.yMin <= b.yMax and a.yMax >= b.yMin and a.zMin <= b.zMax and a.zMax >= b.zMin)

    def intersect_aabb_v_point(aabb, point):
        return (point.x >= aabb.xMin and point.x <= aabb.xMax and point.y >= aabb.yMin and point.y <= aabb.yMax and point.z >= aabb.zMin and point.z <= aabb.zMax)

    def intersect_sphere_v_point(point, sphere):
        # we are using multiplications because is faster than calling Math.pow
        distance = np.sqrt((point.x - sphere.x) * (point.x - sphere.x) + (point.y - sphere.y) * (point.y - sphere.y) + (point.z - sphere.z) * (point.z - sphere.z))
        return distance <= sphere.r

    def intersect_sphere_v_aabb(aabb, sphere):
        # get box closest point to sphere center by clamping
        x = max(aabb.xMin, min(sphere.x, aabb.xMax))
        y = max(aabb.yMin, min(sphere.y, aabb.yMax))
        z = max(aabb.zMin, min(sphere.z, aabb.zMax))

        # this is the same as isPointInsideSphere
        distance = np.sqrt((x - sphere.x) * (x - sphere.x) + (y - sphere.y) * (y - sphere.y) + (z - sphere.z) * (z - sphere.z))
        return distance <= sphere.r

    def intersect_point_v_point_3d(pointA, pointB):
        x = pointA.x - pointB.x
        y = pointA.y - pointB.y
        z = pointA.z - pointB.z

        dist = np.array([x, y, z])
        d = np.linalg.norm(dist)

        if d == 0:
            return True
        else:
            return False

    def intersect_triangle_v_point(trig, point):
        x1 = trig.vertixAX
        x2 = trig.vertixBX
        x3 = trig.vertixCX
        y1 = trig.vertixAY
        y2 = trig.vertixBY
        y3 = trig.vertixCY

        px = point.x
        py = point.y

        areaOrig = abs((x2-x1) * (y3-y1) - (x3-x1) * (y2-y1))
        area1 = abs((x1-px) * (y2-py) - (x2-px) * (y1-py))
        area2 = abs((x2-px) * (y3-py) - (x3-px) * (y2-py))
        area3 = abs((x3-px) * (y1-py) - (x1-px) * (y3-py))

        if area1 + area2 + area3 == areaOrig:
            return True
        else:
            return False

    def __trig_area(A, B, C):
        Ax = A[0, 0]
        Ay = A[1, 0]
        Bx = B[0, 0]
        By = B[1, 0]
        Cx = C[0, 0]
        Cy = C[1, 0]

        AB = np.array([[Bx - Ax], [By - Ay]])
        AC = np.array([[Cx - Ax], [Cy - Ay]])

        cross_prod = np.cross(np.transpose(AB), np.transpose(AC))
        return abs(cross_prod) / 2

    def intersect_line_v_circle(radius, O, P, Q):
        distPQ = P - Q
        minimumDistance = 2 * ShapeCollision.__trig_area(O, P, Q) / np.linalg.norm(distPQ)

        if minimumDistance <= radius:
            return True
        else:
            return False

    def intersect_rectangle_v_rectangle(rec1, rec2):
        r1x = rec1.x
        r1y = rec1.y
        r1w = rec1.w
        r1h = rec1.h

        r2x = rec2.x
        r2y = rec2.y
        r2w = rec2.w
        r2h = rec2.h

        if ((r1x + r1w >= r2x) and (r1x <= r2x + r2w) and (r1y + r1h >= r2y) and (r1y <= r2y + r2h)):
            return True
        else:
            return False

    def intersect_line_v_line(line1, line2):
        x1 = line1.xs
        y1 = line1.ys
        x2 = line1.xe
        y2 = line1.ye

        x3 = line2.xs
        y3 = line2.ys
        x4 = line2.xe
        y4 = line2.ye

        uA = ((x4-x3) * (y1-y3) - (y4-y3) * (x1-x3)) / ((y4-y3) * (x2-x1) - (x4-x3) * (y2-y1))
        uB = ((x2-x1) * (y1-y3) - (y2-y1) * (x1-x3)) / ((y4-y3) * (x2-x1) - (x4-x3) * (y2-y1))

        if uA >= 0 and uA <= 1 and uB >= 0 and uB <= 1:
            return True
        return False

    def intersect_line_v_rectangle(line, rec):
        v1 = np.array([0, 0]).reshape(2, 1)
        v2 = np.array([rec.w, 0]).reshape(2, 1)
        v3 = np.array([rec.w, rec.h]).reshape(2, 1)
        v4 = np.array([0, rec.h]).reshape(2, 1)

        v1 = rbt.rot2d(rec.angle) @ v1 + np.array([rec.x, rec.y]).reshape(2, 1)
        v2 = rbt.rot2d(rec.angle) @ v2 + np.array([rec.x, rec.y]).reshape(2, 1)
        v3 = rbt.rot2d(rec.angle) @ v3 + np.array([rec.x, rec.y]).reshape(2, 1)
        v4 = rbt.rot2d(rec.angle) @ v4 + np.array([rec.x, rec.y]).reshape(2, 1)

        l1 = ShapeLine2D(v1[0, 0], v1[1, 0], v2[0, 0], v2[1, 0])
        l2 = ShapeLine2D(v2[0, 0], v2[1, 0], v3[0, 0], v3[1, 0])
        l3 = ShapeLine2D(v3[0, 0], v3[1, 0], v4[0, 0], v4[1, 0])
        l4 = ShapeLine2D(v4[0, 0], v4[1, 0], v1[0, 0], v1[1, 0])

        if (ShapeCollision.intersect_line_v_line(line, l1) or ShapeCollision.intersect_line_v_line(line, l2) or ShapeCollision.intersect_line_v_line(line, l3) or
                ShapeCollision.intersect_line_v_line(line, l4)):
            return True
        else:
            return False

    def intersect_point_v_rectangle(point, rec):
        rx = rec.x
        ry = rec.y
        rw = rec.w
        rh = rec.h

        px = point.x
        py = point.y

        if (px >= rx) and (px <= rx + rw) and (py >= ry) and (py <= ry + rh):
            return True
        else:
            return False

    def intersect_circle_v_rectangle(circle, rec):
        deltaX = circle.x - max(rec.x, min(circle.x, rec.x + rec.w))
        deltaY = circle.y - max(rec.y, min(circle.y, rec.y + rec.h))
        return (deltaX*deltaX + deltaY*deltaY) < (circle.r * circle.r)

    def intersect_ellipse_v_point(ellipse, point):
        ellipsoidCenter = np.array([ellipse.x, ellipse.y]).reshape(2, 1)
        ellipsoidAxis = np.array([ellipse.a, ellipse.b]).reshape(2, 1)
        pointCheck = np.array([point.x, point.y]).reshape(2, 1) - ellipsoidCenter
        pointCheckRotateBack = ellipse.rotMat.T @ pointCheck
        mid = pointCheckRotateBack / ellipsoidAxis
        midsq = mid**2
        eq = sum(midsq)
        if eq <= 1.0:
            return True
        else:
            return False


if __name__ == "__main__":

    def triangle_and_point():
        c = ShapePoint2D(5, 5)
        trig = ShapeTriangle([0, 0], [10, 0], [0, 10])
        collision = ShapeCollision.intersect_triangle_v_point(trig, c)
        print(f"> collision: {collision}")


    def line_and_circle():
        r = 1
        O = np.array([[0], [0]])
        P = np.array([[-6], [-6]])
        Q = np.array([[-6], [6]])
        collision = ShapeCollision.intersect_line_v_circle(r, O, P, Q)
        print(f"> collision: {collision}")


    def rectangle_and_rectangle():
        rec1 = ShapeRectangle(0, 0, 5, 5)
        rec2 = ShapeRectangle(10, 10, 5, 5)
        collision = ShapeCollision.intersect_rectangle_v_rectangle(rec1, rec2)
        print(f"> collision: {collision}")


    def line_and_rectangle():
        recWithAngle = ShapeRectangle(1,1,1,5,angle=2)
        line = ShapeLine2D(-1,1,0,5)
        collision = ShapeCollision.intersect_line_v_rectangle(line, recWithAngle)
        print(f"> collision: {collision}")
        line.plot()
        recWithAngle.plot()
        plt.show()


    def ellipsoid():
        s = ShapeSuperEllipsoid2D(a=10, b=10, n=4)
        s.plot()
        plt.show()


    def ellipse_and_point():
        theta = 1.0
        rM = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        ellipse = ShapeEllipse(0, 4, 5, 3, rM)
        p1 = ShapePoint2D(4.58, 1.14)
        p2 = ShapePoint2D(0.0, 0.0)
        p3 = ShapePoint2D(4.0, 3.8)
        p4 = ShapePoint2D(-2.5, 6.4)
        state1 = ShapeCollision.intersect_ellipse_v_point(ellipse,p1); print(f"> state1: \n{state1}")
        state2 = ShapeCollision.intersect_ellipse_v_point(ellipse,p2); print(f"> state2: \n{state2}")
        state3 = ShapeCollision.intersect_ellipse_v_point(ellipse,p3); print(f"> state3: \n{state3}")
        state4 = ShapeCollision.intersect_ellipse_v_point(ellipse,p4); print(f"> state4: \n{state4}")

        ellipse.plot()
        p1.plot()
        p2.plot()
        p3.plot()
        p4.plot()
        plt.show()