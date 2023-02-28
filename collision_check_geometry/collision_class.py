# ∧ is exactly 'and' in this context. ∨ means 'or'. 
# You can notice the similarity both in form and meaning with ∩ and ∪ from set theory.
# https://developer.mozilla.org/en-US/docs/Games/Techniques/3D_collision_detection
import math
import numpy as np

class aabb_obj:
    def __init__(self,x_min,y_min,z_min,x_max,y_max,z_max):
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max

class sqr_rec_2d_obj:
    def __init__(self,x,y,h,w):
        self.x = x
        self.y = y
        self.h = h
        self.w = w
        
class point3d_obj:
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z

class point2d_obj:
    def __init__(self,x,y):
        self.x = x
        self.y = y

class sphere_obj:
    def __init__(self,x,y,z,r):
        self.x = x
        self.y = y
        self.z = z
        self.r = r

class circle_obj:
     def __init__(self,x,y,r):
        self.x = x
        self.y = y
        self.r = r

class triangle_obj:
    def __init__(self,vertix_a,vertix_b,vertix_c):
        self.vertix_a_x = vertix_a[0]
        self.vertix_a_y = vertix_a[1]
        self.vertix_b_x = vertix_b[0]
        self.vertix_b_y = vertix_b[1]
        self.vertix_c_x = vertix_c[0]
        self.vertix_c_y = vertix_c[1]
        
def intersect_aabb_v_aabb(a, b):
    return a.x_min <= b.x_max and \
    a.x_max >= b.x_min and \
    a.y_min <= b.y_max and \
    a.y_max >= b.y_min and \
    a.z_min <= b.z_max and \
    a.z_max >= b.z_min

def intersect_aabb_v_point(aabb,point):
    return point.x >= aabb.x_min and \
    point.x <= aabb.x_max and \
    point.y >= aabb.y_min and \
    point.y <= aabb.y_max and \
    point.z >= aabb.z_min and \
    point.z <= aabb.z_max

def intersect_sphere_v_point(point, sphere):
    # we are using multiplications because is faster than calling Math.pow
    distance = math.sqrt((point.x - sphere.x) * (point.x - sphere.x) + \
                         (point.y - sphere.y) * (point.y - sphere.y) + \
                         (point.z - sphere.z) * (point.z - sphere.z))
    return distance <= sphere.r

def intersect_sphere_v_aabb(aabb, sphere):
    # get box closest point to sphere center by clamping
    x = max(aabb.x_min, min(sphere.x, aabb.x_max))
    y = max(aabb.y_min, min(sphere.y, aabb.y_max))
    z = max(aabb.z_min, min(sphere.z, aabb.z_max))

    # this is the same as isPointInsideSphere
    distance = math.sqrt((x - sphere.x) * (x - sphere.x) + (y - sphere.y) * (y - sphere.y) + (z - sphere.z) * (z - sphere.z))
    return distance <= sphere.r

def point_to_point_3d(point_a,point_b):
    x = point_a.x - point_b.x
    y = point_a.y - point_b.y
    z = point_a.z - point_b.z

    dist = np.array([x,y,z])
    d = np.linalg.norm(dist)

    if d == 0:
        return True
    else:
        return False

def triangle_to_point(trig,point):
    # http://www.jeffreythompson.org/collision-detection/tri-point.php

    x1 = trig.vertix_a_x
    x2 = trig.vertix_b_x
    x3 = trig.vertix_c_x
    y1 = trig.vertix_a_y
    y2 = trig.vertix_b_y
    y3 = trig.vertix_c_y

    px = point.x
    py = point.y

    areaOrig = abs( (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1) )
    area1 =    abs( (x1-px)*(y2-py) - (x2-px)*(y1-py) )
    area2 =    abs( (x2-px)*(y3-py) - (x3-px)*(y2-py) )
    area3 =    abs( (x3-px)*(y1-py) - (x1-px)*(y3-py) )

    if (area1+area2+area3== areaOrig):
        return True
    else:
        return False

def trig_area(A,B,C):
    Ax = A[0,0]
    Ay = A[1,0]

    Bx = B[0,0]
    By = B[1,0]

    Cx = C[0,0]
    Cy = C[1,0]

    AB = np.array([[Bx - Ax],
                  [By - Ay]])

    AC = np.array([[Cx - Ax],
                  [Cy - Ay]])

    cross_prod = np.cross(np.transpose(AB),np.transpose(AC)) # cross prod is scalar AB[0,0]*AC[1,0] - AB[1,0]*AC[0,0]

    return abs(cross_prod)/2

def line_circle_collision(radius,O,P,Q): # https://www.baeldung.com/cs/circle-line-segment-collision-detection

    distPQ = P - Q
    minimum_distance = 2*trig_area(O,P,Q)/np.linalg.norm(distPQ)

    if minimum_distance <= radius:
        return True
    else:
        return False

def intersect_rectangle_v_rectangle(rec1,rec2):
    r1x = rec1.x
    r1y = rec1.y
    r1w = rec1.w
    r1h = rec1.h

    r2x = rec2.x
    r2y = rec2.y
    r2w = rec2.w
    r2h = rec2.h

    if (r1x + r1w >= r2x) and (r1x <= r2x + r2w) and (r1y + r1h >= r2y) and (r1y <= r2y + r2h) :
        return True
    else:
        return False
