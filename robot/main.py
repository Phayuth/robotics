from ur5e import ur5e
from planar_rr import planar_rr
from planar_rrr import planar_rrr
from spatial_rr import spatial_rr
from spatial_rrp import spatial_rrp
import numpy as np

# Test
r = ur5e()
a = np.array([[0], [0], [0], [0], [0], [0]])
current = r.forward_kinematic(a)
Jac = r.jacobian(a)
print(current)
print(Jac)