from gekko import GEKKO
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
from scipy.spatial import HalfspaceIntersection
from scipy.optimize import linprog
from matplotlib.patches import Circle


def solve_with_scipy():
    # show halfspaces
    x = np.linspace(-1, 4, 100)
    y = np.linspace(-1, 4, 100)
    X, Y = np.meshgrid(x, y)

    def fx1(x, y):
        return 2 * x + y - 4 <= 0

    def fx2(x, y):
        return -0.5 * x + y - 2 <= 0

    def fx3(x, y):
        return -x + 0 * y <= 0

    def fx4(x, y):
        return 0 * x - y <= 0

    def ff1(x):
        return -2 * x + 4

    def ff2(x):
        return 0.5 * x + 2

    fig, axes = plt.subplots(1, 2)
    FX1 = fx1(X, Y)
    FX2 = fx2(X, Y)
    FX3 = fx3(X, Y)
    FX4 = fx4(X, Y)
    axes[0].plot(X[FX1], Y[FX1], "b*")
    axes[0].plot(X[FX2], Y[FX2], "r.")
    axes[0].plot(X[FX3], Y[FX3], "g+")
    axes[0].plot(X[FX4], Y[FX4], "k,")
    axes[0].plot(x, ff1(x))
    axes[0].plot(x, ff2(x))
    axes[0].axhline()
    axes[0].axvline()
    axes[0].set_xlim(-1, 4)
    axes[0].set_ylim(-1, 4)
    axes[0].set_title("Halfspaces")
    axes[0].set_aspect("equal")
    merge = FX1 & FX2 & FX3 & FX4
    axes[1].plot(X[merge], Y[merge], "b*")
    axes[1].plot(X[merge], Y[merge], "r.")
    axes[1].plot(X[merge], Y[merge], "g+")
    axes[1].plot(X[merge], Y[merge], "k,")
    axes[1].plot(x, ff1(x))
    axes[1].plot(x, ff2(x))
    axes[1].axhline()
    axes[1].axvline()
    axes[1].set_xlim(-1, 4)
    axes[1].set_ylim(-1, 4)
    axes[1].set_title("Intersection of Halfspaces")
    axes[1].set_aspect("equal")
    plt.show()

    halfspaces = np.array(
        [
            [-1.0, 0.0, 0.0],  # -x + 0y <= 0
            [0.0, -1.0, 0.0],  # 0x -y <= 0
            [2.0, 1.0, -4.0],  # 2x +y <= 4
            [-0.5, 1.0, -2.0],  # -0.5x +y <= 2
        ]
    )

    feasible_point = np.array([0.5, 0.5])  # Interior Point: Must be strictly within the feasible region; otherwise, the algorithm won't work. init guess

    hs = HalfspaceIntersection(halfspaces, feasible_point)
    xintsect, yintsect = zip(*hs.intersections)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    xlim, ylim = (-1, 3), (-1, 3)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    x = np.linspace(-1, 3, 100)
    symbols = ["-", "+", "x", "*"]
    signs = [0, 0, -1, -1]
    fmt = {"color": None, "edgecolor": "b", "alpha": 0.5}

    for h, sym, sign in zip(halfspaces, symbols, signs):
        hlist = h.tolist()
        fmt["hatch"] = sym
        if h[1] == 0:
            ax.axvline(-h[2] / h[0], label="{}x+{}y+{}=0".format(*hlist))
            xi = np.linspace(xlim[sign], -h[2] / h[0], 100)
            ax.fill_between(xi, ylim[0], ylim[1], **fmt)
        else:
            ax.plot(x, (-h[2] - h[0] * x) / h[1], label="{}x+{}y+{}=0".format(*hlist))
            ax.fill_between(x, (-h[2] - h[0] * x) / h[1], ylim[sign], **fmt)

    ax.plot(xintsect, yintsect, "o", markersize=8)

    # determine interior circle
    norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1), (halfspaces.shape[0], 1))
    c = np.zeros((halfspaces.shape[1],))
    c[-1] = -1
    A = np.hstack((halfspaces[:, :-1], norm_vector))
    b = -halfspaces[:, -1:]
    res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
    x = res.x[:-1]
    y = res.x[-1]

    circle = Circle(x, radius=y, alpha=0.3)
    ax.add_patch(circle)
    plt.legend(bbox_to_anchor=(1.6, 1.0))
    plt.show()


def solve_with_gekko():
    # set initial parameters
    xd = 3
    yd = 1
    xinit = 0.5
    yinit = 0.5

    m = GEKKO()
    x = m.Var(value=xinit)
    y = m.Var(value=yinit)

    m.Obj((xd - x) ** 2 + (yd - y) ** 2)  # minimize square cost
    # m.Obj(-((xd - x) ** 2 + (yd - y) ** 2)) # maximize square cost

    # halfspace equations
    m.Equations(
        [
            2 * x + y - 4 <= 0,
            -0.5 * x + y - 2 <= 0,
            -x + 0 * y <= 0,
            0 * x - y <= 0,
        ]
    )
    m.solve(disp=False)
    print([x.value, y.value])

    xi = np.linspace(-1, 4, 100)

    def hs1(x):
        return -2 * x + 4

    def hs2(x):
        return 0.5 * x + 2

    fig, ax = plt.subplots()
    ax.plot(xi, hs1(xi))
    ax.plot(xi, hs2(xi))
    ax.axhline()
    ax.axvline()
    ax.plot(xd, yd, "r*", label="desired pose")
    ax.plot(xinit, yinit, "b^", label="initial guess pose")
    ax.plot(x.value[0], y.value[0], "g+", label="optimizer pose")
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.legend()
    plt.show()


def solve_with_cvxpy():
    # Halfspaces in the form: a*x + b*y <= c
    halfspaces = np.array(
        [
            [-1.0, 0.0, 0.0],  # -x ≤ 0 → x ≥ 0
            [0.0, -1.0, 0.0],  # -y ≤ 0 → y ≥ 0
            [2.0, 1.0, 4.0],  # 2x + y ≤ 4
            [-0.5, 1.0, 2.0],  # -0.5x + y ≤ 2
        ]
    )

    p0 = np.array([3.0, 3.0])
    p = cp.Variable(2)
    objective = cp.Minimize(cp.sum_squares(p - p0))
    # Constraints: a.T @ p <= c for each halfspace [a, b, c]
    constraints = [halfspaces[i, :2] @ p <= halfspaces[i, 2] for i in range(halfspaces.shape[0])]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    popt = p.value
    print(f"> popt: {popt}")

    fig, ax = plt.subplots()
    x_vals = np.linspace(-1, 5, 400)
    for a, b, c in halfspaces:
        if abs(b) > 1e-6:
            y_vals = (c - a * x_vals) / b
            ax.plot(x_vals, y_vals, "k--")
        else:
            x_line = c / a
            ax.axvline(x_line, linestyle="--", color="k")
    ax.plot(p0[0], p0[1], "ro", label="Fixed point")
    ax.plot(popt[0], popt[1], "go", label="Closest point")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # solve_with_scipy()
    # solve_with_gekko()
    solve_with_cvxpy()
