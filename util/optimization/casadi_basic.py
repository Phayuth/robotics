from casadi import *
from icecream import ic

x = MX.sym("x")  # 1by1 matrix
ic(x)

y = SX.sym('y', 5)  # 5 elements vector
ic(y)

Z = SX.sym('Z', 4, 2)  # 4x2 matrix
ic(Z)

# SX.sym is a (static) function which returns an SX instance. When variables have been declared, expressions can now be formed in an intuitive way:

f = x**2 + 10
f = sqrt(f)

ic(f)

# constant matrix, if we see @1 it is = 0
# if we see 00 = it is structural zero which that is suppose to be zero, while 0 is variable 0.

B1 = SX.zeros(4, 5)  #: A dense 4-by-5 empty matrix with all zeros
B2 = SX(4, 5)  #:  A sparse 4-by-5 empty matrix with all zeros
B4 = SX.eye(4)  #: A sparse 4-by-4 matrix with ones on the diagonal
ic(B4)