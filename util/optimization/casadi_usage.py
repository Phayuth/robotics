import casadi as ca
from icecream import ic


def building_block():
    x = ca.MX.sym("x")       # 1x1 matrix
    ic(x)
    y = ca.SX.sym('y', 5)    # 5 elements vector
    ic(y)
    Z = ca.SX.sym('Z', 4, 2) # 4x2 matrix
    ic(Z)

    # SX.sym is a (static) function which returns an SX instance. When variables have been declared, expressions can now be formed in an intuitive way:
    f = x**2 + 10
    f = ca.sqrt(f)
    ic(f)

    # constant matrix, if we see @1 it is = 0
    # if we see 00 = it is structural zero which that is suppose to be zero, while 0 is variable 0.

    B1 = ca.SX.zeros(4, 5)  # A dense 4x5 empty matrix with all zeros
    B2 = ca.SX(4, 5)        # A sparse 4x5 empty matrix with all zeros
    B4 = ca.SX.eye(4)       # A sparse 4x4 matrix with ones on the diagonal
    ic(B4)


def simple_problem1():
    # Symbols/expressions
    x = ca.MX.sym('x')
    y = ca.MX.sym('y')
    z = ca.MX.sym('z')
    f = x**2 + 100 * z**2
    g = z + (1 - x)**2 - y

    # NLP problem setup
    nlp = {}
    nlp['x'] = ca.vertcat(x, y, z)  # decision vars
    nlp['f'] = f                    # objective
    nlp['g'] = g                    # constraints

    # Create solver instance
    solver = ca.nlpsol('F', 'ipopt', nlp)

    # Solve the problem using a guess
    res = solver(x0=[2.5, 3.0, 0.75], ubg=0, lbg=0)

    # Response
    ic(res['x'])
    resAsNpArray = res['x'].__array__()
    ic(resAsNpArray)


def simple_problem2():
    x = ca.MX.sym('t')
    f = x**2

    nlp = {}
    nlp['x'] = ca.vertcat(x)
    nlp['f'] = f
    solver = ca.nlpsol('F', 'ipopt', nlp)
    solution = solver(x0=[2.3], ubg=0.0, lbg=10.0)
    ic(solution['x'])


def simple_problem3():
    x = ca.MX.sym('x')
    y = ca.MX.sym('y')
    objective = x**2 + y**2

    optProblem = {'x': ca.vertcat(x, y), 'f': objective}
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-6}
    solver = ca.nlpsol('solver', 'ipopt', optProblem, opts)
    InitialGuess = [1.0, 2.0]
    solution = solver(x0=InitialGuess)

    ic(solution['x'])


if __name__ == "__main__":
    # simple_problem1()
    # simple_problem2()
    simple_problem3()