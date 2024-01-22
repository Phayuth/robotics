from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt


def optimal_control_intergral_obj():
    m = GEKKO()

    nt = 101
    m.time = np.linspace(0, 2, nt)

    x1 = m.Var(value=1)
    x2 = m.Var(value=0)
    # u = m.Var(value=0, lb=-1, ub=1)
    u = m.MV(value=0, lb=-1, ub=1)
    u.STATUS = 1

    p = np.zeros(nt)  # mark final time point
    p[-1] = 1.0
    final = m.Param(value=p)

    # Equations
    m.Equation(x1.dt() == u)
    m.Equation(x2.dt() == (1/2) * x1**2)
    m.Obj(x2 * final)  # Objective function

    m.options.IMODE = 6  # optimal control mode
    m.solve(disp=True)  # solve

    plt.figure(1)  # plot results
    plt.plot(m.time, x1.value, 'k-', label=r'$x_1$')
    plt.plot(m.time, x2.value, 'b-', label=r'$x_2$')
    plt.plot(m.time, u.value, 'r--', label=r'$u$')
    plt.legend(loc='best')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()


def optimal_control_free_terminal_time():  # Free terminal time, integral objective and differential equations as constraints
    m = GEKKO()

    n = 501
    tm = np.linspace(0, 1, n)
    m.time = tm

    # Variables
    x1 = m.Var(value=1)
    x2 = m.Var(value=2)
    u = m.MV(value=-1, fixed_initial=False)
    u.STATUS = 1
    u.DCOST = 1e-5

    p = np.zeros(n)
    p[-1] = 1.0
    final = m.Param(value=p)

    # FV
    tf = m.FV(value=10.0, lb=0.0, ub=100.0)
    tf.STATUS = 1

    # equations
    m.Equation(x1.dt() / tf == x2)
    m.Equation(x2.dt() / tf == u)

    # Final conditions
    soft = True
    if soft:
        # soft terminal constraint
        m.Minimize(final * 1e5 * (x1 - 3)**2)
        # m.Minimize(final*1e5*(x2-2)**2)
    else:
        # hard terminal constraint
        x1f = m.Param()
        m.free(x1f)
        m.fix_final(x1f, 3)
        # connect endpoint parameters to x1 and x2
        m.Equations([x1f == x1])

    # Objective Function
    obj = m.Intermediate(m.integral((1/2) * u**2))

    m.Minimize(final * obj)

    m.options.IMODE = 6
    m.options.NODES = 3
    m.options.SOLVER = 3
    m.options.MAX_ITER = 500
    # m.options.MV_TYPE = 0
    m.options.DIAGLEVEL = 0
    m.solve(disp=True)

    # Create a figure
    tm = tm * tf.value[0]
    plt.figure(figsize=(10, 4))
    plt.subplot(2, 2, 1)
    # plt.plot([0,1],[1/9,1/9],'r:',label=r'$x<\frac{1}{9}$')
    plt.plot(tm, x1.value, 'k-', lw=2, label=r'$x1$')
    plt.ylabel('x1')
    plt.legend(loc='best')
    plt.subplot(2, 2, 2)
    plt.plot(tm, x2.value, 'b--', lw=2, label=r'$x2$')
    plt.ylabel('x2')
    plt.legend(loc='best')
    plt.subplot(2, 2, 3)
    plt.plot(tm, u.value, 'r--', lw=2, label=r'$u$')
    plt.ylabel('control')
    plt.legend(loc='best')
    plt.xlabel('Time')
    plt.subplot(2, 2, 4)
    plt.plot(tm, obj.value, 'g-', lw=2, label=r'$\frac{1}{2} \int u^2$')
    plt.text(0.5, 3.0, 'Final Value = ' + str(np.round(obj.value[-1], 2)))
    plt.ylabel('Objective')
    plt.legend(loc='best')
    plt.xlabel('Time')
    plt.show()


def optimal_control_minimum_flight_time():
    m = GEKKO()

    nt = 101
    tm = np.linspace(0, 1, nt)
    m.time = tm

    x1 = m.Var(value=-2.5, lb=-100, ub=100)
    x2 = m.Var(value=0, lb=-100, ub=100)
    u = m.MV(value=0, lb=-np.pi, ub=np.pi)
    u.STATUS = 1
    u.DCOST = 0.1

    p = np.zeros(nt)
    p[-1] = 1.0
    final = m.Param(value=p)

    tf = m.FV(value=10, lb=0.1, ub=100.0)
    tf.STATUS = 1

    c = m.if3(x2 - 1, 1, (x2 - 1)**2 + 1) # if x2 <= 1 then 1, if x2 > 1 then (x2-1)**1 + 1
    m.Equation(x1.dt() == c * m.cos(u) * tf)
    m.Equation(x2.dt() == c * m.sin(u) * tf)

    # hard constraints (fix endpoint)
    #m.fix_final(x1,3)
    #m.fix_final(x2,0)

    # soft constraints (objective)
    m.Minimize(100 * final * (x1 - 3)**2)
    m.Minimize(100 * final * (x2 - 0)**2)

    # minimize final time
    # initialize with IPOPT Solver
    m.Minimize(tf)
    m.options.IMODE = 6
    m.options.SOLVER = 3
    m.solve()

    # find MINLP solution with APOPT Solver
    m.options.SOLVER = 1
    m.options.TIME_SHIFT = 0
    m.solve()

    tm = tm * tf.value[0]

    plt.figure(figsize=(8, 5))
    plt.plot(tm, x1.value, 'k-', lw=2, label=r'$x_1$')
    plt.plot(tm, x2.value, 'b-', lw=2, label=r'$x_2$')
    plt.plot(tm, u.value, 'r--', lw=2, label=r'$u$')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()


if __name__ == "__main__":
    # optimal_control_intergral_obj()
    # optimal_control_free_terminal_time()
    optimal_control_minimum_flight_time()