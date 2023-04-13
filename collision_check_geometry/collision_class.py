""" Collision check class based on geometry

∧ is exactly 'and' in this context. ∨ means 'or'. You can notice the similarity both in form and meaning with ∩ and ∪ from set theory.
References:
    - https://developer.mozilla.org/en-US/docs/Games/Techniques/3D_collision_detection
    - http://www.jeffreythompson.org/collision-detection/table_of_contents.php
    - https://www.baeldung.com/cs/circle-line-segment-collision-detection

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt


class ObjAabb:

    def __init__(self, x_min, y_min, z_min, x_max, y_max, z_max):
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max


class ObjRec:

    def __init__(self, x, y, h, w, p=None):
        self.x = x
        self.y = y
        self.h = h
        self.w = w
        self.p = p  # probability value

    def plot(self):
        if self.p == 1:
            color = "r"
        else:
            color = "b"
        plt.plot([self.x, self.x + self.w], [self.y, self.y], c=color)
        plt.plot([self.x, self.x], [self.y, self.y + self.h], c=color)
        plt.plot([self.x, self.x + self.w], [self.y + self.h, self.y + self.h], c=color)
        plt.plot([self.x + self.w, self.x + self.w], [self.y, self.y + self.h], c=color)


class ObjPoint3D:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class ObjPoint2D:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def plot(self):
        plt.scatter(self.x, self.y)


class ObjLine2D:

    def __init__(self, xs, ys, xe, ye):
        self.xs = xs
        self.ys = ys
        self.xe = xe
        self.ye = ye

    def plot(self):
        plt.plot([self.xs, self.xe], [self.ys, self.ye])


class ObjSphere:

    def __init__(self, x, y, z, r):
        self.x = x
        self.y = y
        self.z = z
        self.r = r


class ObjCircle:

    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r


class ObjTriangle:

    def __init__(self, vertix_a, vertix_b, vertix_c):
        self.vertix_a_x = vertix_a[0]
        self.vertix_a_y = vertix_a[1]
        self.vertix_b_x = vertix_b[0]
        self.vertix_b_y = vertix_b[1]
        self.vertix_c_x = vertix_c[0]
        self.vertix_c_y = vertix_c[1]


def intersect_aabb_v_aabb(a, b):
    return (a.x_min <= b.x_max and a.x_max >= b.x_min and a.y_min <= b.y_max and a.y_max >= b.y_min and a.z_min <= b.z_max and a.z_max >= b.z_min)


def intersect_aabb_v_point(aabb, point):
    return (point.x >= aabb.x_min and point.x <= aabb.x_max and point.y >= aabb.y_min and point.y <= aabb.y_max and point.z >= aabb.z_min and point.z <= aabb.z_max)


def intersect_sphere_v_point(point, sphere):
    # we are using multiplications because is faster than calling Math.pow
    distance = np.sqrt((point.x - sphere.x) * (point.x - sphere.x) + (point.y - sphere.y) * (point.y - sphere.y) + (point.z - sphere.z) * (point.z - sphere.z))
    return distance <= sphere.r


def intersect_sphere_v_aabb(aabb, sphere):
    # get box closest point to sphere center by clamping
    x = max(aabb.x_min, min(sphere.x, aabb.x_max))
    y = max(aabb.y_min, min(sphere.y, aabb.y_max))
    z = max(aabb.z_min, min(sphere.z, aabb.z_max))

    # this is the same as isPointInsideSphere
    distance = np.sqrt((x - sphere.x) * (x - sphere.x) + (y - sphere.y) * (y - sphere.y) + (z - sphere.z) * (z - sphere.z))
    return distance <= sphere.r


def intersect_point_v_point_3d(point_a, point_b):
    x = point_a.x - point_b.x
    y = point_a.y - point_b.y
    z = point_a.z - point_b.z

    dist = np.array([x, y, z])
    d = np.linalg.norm(dist)

    if d == 0:
        return True
    else:
        return False


def intersect_triangle_v_point(trig, point):
    x1 = trig.vertix_a_x
    x2 = trig.vertix_b_x
    x3 = trig.vertix_c_x
    y1 = trig.vertix_a_y
    y2 = trig.vertix_b_y
    y3 = trig.vertix_c_y

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


def trig_area(A, B, C):
    Ax = A[0, 0]
    Ay = A[1, 0]

    Bx = B[0, 0]
    By = B[1, 0]

    Cx = C[0, 0]
    Cy = C[1, 0]

    AB = np.array([[Bx - Ax], [By - Ay]])

    AC = np.array([[Cx - Ax], [Cy - Ay]])

    cross_prod = np.cross(np.transpose(AB), np.transpose(AC))
    # cross prod is scalar AB[0,0]*AC[1,0] - AB[1,0]*AC[0,0]

    return abs(cross_prod) / 2


def intersect_line_v_circle(radius, O, P, Q):
    distPQ = P - Q
    minimum_distance = 2 * trig_area(O, P, Q) / np.linalg.norm(distPQ)

    if minimum_distance <= radius:
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
    l1 = ObjLine2D(rec.x, rec.y, rec.x + rec.w, rec.y)
    l2 = ObjLine2D(rec.x, rec.y, rec.x, rec.y + rec.h)
    l3 = ObjLine2D(rec.x, rec.y + rec.h, rec.x + rec.w, rec.y + rec.h)
    l4 = ObjLine2D(rec.x + rec.w, rec.y, rec.x + rec.w, rec.y + rec.h)

    if (intersect_line_v_line(line, l1) or intersect_line_v_line(line, l2) or intersect_line_v_line(line, l3) or intersect_line_v_line(line, l4)):
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
