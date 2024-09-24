import numpy as np
import scipy.optimize as sop
import matplotlib.pyplot as plt
from gekko import GEKKO


# https://github.com/MatthewPeterKelly/OptimTraj/blob/master/demo/cartPole/MAIN_minForce.m

# physical prop
l = 0.5  # m
m1 = 2.0  # kg
m2 = 0.5  # kg
g = 9.81  # m/s2
dmax = 10  # m
umax = 100  # newton force

m = GEKKO(remote=True)
tf = 2.0 # fixed final time
nt = 501
m.time = np.linspace(0, tf, nt)

# Variables
q1 = m.Var(value=0.0, lb=-dmax, ub=dmax)
q2 = m.Var(value=0.0)
q1d = m.Var(value=0.0)
q2d = m.Var(value=0.0)

# optimized control var
u = m.MV(value=0, lb=-umax, ub=umax)
u.STATUS = 1

# constraint
m.Equation(q1.dt() == q1d)
m.Equation(q2.dt() == q2d)
m.Equation(q1d.dt() == (l * m2 * m.sin(q2) * q2d**2 + u + m2 * g * m.cos(q2) * m.sin(q2)) / (m1 + m2 * (1 - m.cos(q2) ** 2)))
m.Equation(q2d.dt() == -(l * m2 * m.cos(q2) * m.sin(q2) * q2d**2 + u * m.cos(q2) + (m1 + m2) * g * m.sin(q2)) / (l * m1 + l * m2 * (1 - m.cos(q2) ** 2)))

# define boundary constraints (dont defined initial condition with fix, define when create the variable)
# m.fix(q1, pos=0, val=0.0)
# m.fix(q2, pos=0, val=0.0)
# m.fix(q1d, pos=0, val=0.0)
# m.fix(q2d, pos=0, val=0.0)

d = 1.0  # m
m.fix(q1, pos=len(m.time) - 1, val=d)
m.fix(q2, pos=len(m.time) - 1, val=np.pi)
m.fix(q1d, pos=len(m.time) - 1, val=0.0)
m.fix(q2d, pos=len(m.time) - 1, val=0.0)

# Objective function
m.Obj(u**2)

# Solve
m.options.IMODE = 6
m.solve(disp=True)

fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

axs[0].plot(m.time, q1.VALUE, "ro")
axs[0].set_ylabel(f"Position [m]")

axs[1].plot(m.time, q2.VALUE, "go")
axs[1].set_ylabel(f"Angle [rad]")

axs[2].plot(m.time, u.VALUE, "bo")
axs[2].set_ylabel(f"Force [N]")

for i in range(3):
    axs[i].set_xlim(m.time[0], m.time[-1])
    axs[i].grid(True)
axs[-1].set_xlabel("Time")
plt.show()



if False:
    # not correct yet
    NumK = 201
    TF = 2.0
    T = np.linspace(0, TF, NumK)
    Hk = T[1] - T[0]


    def state(xu):
        q1 = xu[0]
        q2 = xu[1]
        q1d = xu[2]
        q2d = xu[3]
        u = xu[4]

        q1dd = (l * m2 * np.sin(q2) * q2d**2 + u + m2 * g * np.cos(q2) * np.sin(q2)) / (m1 + m2 * (1 - np.cos(q2) ** 2))
        q2dd = -(l * m2 * np.cos(q2) * np.sin(q2) * q2d**2 + u * np.cos(q2) + (m1 + m2) * g * np.sin(q2)) / (l * m1 + l * m2 * (1 - np.cos(q2) ** 2))

        return np.array([q1d, q2d, q1dd, q2dd])


    def objective_function(xu):
        xu = np.array(xu).reshape(-1, 5)
        u = xu[:, 4]
        usq = np.square(u)
        vs = np.ones_like(u)
        vs[1:-1] = 2
        element = vs * usq
        J = np.sum(element) * (Hk / 2.0)
        return J


    def startconstr(xu):
        xu = np.array(xu).reshape(-1, 5)
        xk1 = xu[0, 0:4]

        xinit = np.array([0, 0, 0, 0])

        err = xk1 - xinit
        return err


    def endconstr(xu):
        xu = np.array(xu).reshape(-1, 5)
        xkend = xu[0, 0:4]

        d = 1.0 # m desired stop position
        xend = np.array([d, np.pi, 0, 0])

        err = xkend - xend
        return err


    def collocationconstr(xu):
        xu = np.array(xu).reshape(-1, 5)
        x = xu[:, 0:4]
        u = xu[:, 4]

        rhs = np.diff(x, axis=0)

        f = np.array([state(xu[i, :]) for i in range(xu.shape[0])])
        lhs = np.zeros((rhs.shape))
        for i in range(f.shape[0]-1):
            lhs[i] = (f[i] + f[i+1]) * (Hk / 2.0)

        err = lhs - rhs
        colcost = np.linalg.norm(err, axis=1)

        return colcost


    cons = [
        {"type": "eq", "fun": startconstr},
        {"type": "eq", "fun": endconstr},
        {"type": "eq", "fun": collocationconstr},
    ]


    initial_guess = np.array([3,np.pi,0,0,0]*NumK).reshape(-1,5) * T.reshape(-1,1)
    initial_guess = initial_guess.flatten()
    bounds = [(-dmax, dmax), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-umax, umax)]*NumK
    result = sop.minimize(objective_function, initial_guess, bounds=bounds, constraints=cons, method="SLSQP", options={"disp": True})
    print(f"> result: {result}")