import casadi as ca
from icecream import ic


def building_block():
    x = ca.MX.sym("x")  # 1x1 matrix
    ic(x)
    y = ca.SX.sym("y", 5)  # 5 elements vector
    ic(y)
    Z = ca.SX.sym("Z", 4, 2)  # 4x2 matrix
    ic(Z)

    # SX.sym is a (static) function which returns an SX instance. When variables have been declared, expressions can now be formed in an intuitive way:
    f = x**2 + 10
    f = ca.sqrt(f)
    ic(f)

    # constant matrix, if we see @1 it is = 0
    # if we see 00 = it is structural zero which that is suppose to be zero, while 0 is variable 0.

    B1 = ca.SX.zeros(4, 5)  # A dense 4x5 empty matrix with all zeros
    B2 = ca.SX(4, 5)  # A sparse 4x5 empty matrix with all zeros
    B4 = ca.SX.eye(4)  # A sparse 4x4 matrix with ones on the diagonal
    ic(B4)


if __name__ == "__main__":
    building_block()
