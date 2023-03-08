from ur5e import ur5e
from planar_rr import planar_rr
from planar_rrr import planar_rrr
from spatial_rr import spatial_rr
from spatial_rrp import spatial_rrp
import numpy as np

# Test
r = ur5e()

xx = 1
yx = 0
zx = 0

xy = 0
yy = 1
zy = 0

xz = 0
yz = 0
zz = 1

px = 0.3
py = 0.3
pz = 0.3

gd = np.array([[xx, yx, zx, px],
                [xy, yy, zy, py],
                [xz, yz, zz, pz],
                [ 0,  0,  0,  1]]) # this is known by user


# bound joint angle -pi to pi
def bound(theta):
    if theta <= np.pi:
        theta = theta + 2*np.pi
    elif theta > np.pi:
        theta = theta - 2*np.pi
    return theta

# current = r.forward_kinematic(a)
theta_ik = r.inverse_kinematic_geo(gd)
current = r.forward_kinematic(theta_ik)
print("==>> current: ", current)

r.plot_arm(theta_ik)