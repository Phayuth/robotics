# ∧ is exactly 'and' in this context. ∨ means 'or'. 
# You can notice the similarity both in form and meaning with ∩ and ∪ from set theory.
# https://developer.mozilla.org/en-US/docs/Games/Techniques/3D_collision_detection
import math

class aabb_obj:
    def __init__(self,x_min,y_min,z_min,x_max,y_max,z_max):
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max

class point_obj:
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z

class sphere_obj:
    def __init__(self,x,y,z,r):
        self.x = x
        self.y = y
        self.z = z
        self.r = r

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






# Test -------------------------------------------------------------------------
aabb_a = aabb_obj(0.0,0.0,0.0,1.0,1.0,1.0)
aabb_b = aabb_obj(0.5,0.0,0.0,1.0,1.0,1.0)
point = point_obj(0.5,0.0,0.0)
sphere = sphere_obj(0.0,0.0,0.0,4)
print(intersect_aabb_v_aabb(aabb_a,aabb_b))
print(intersect_aabb_v_point(aabb_a,point))
print(intersect_sphere_v_point(point,sphere))
print(intersect_sphere_v_aabb(aabb_a,sphere))