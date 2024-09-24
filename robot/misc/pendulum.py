import os
import sys
sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
import matplotlib.pyplot as plt
from gekko import GEKKO
from simulator.integrator_euler import EulerNumericalIntegrator


class Pendulum:

    def __init__(self) -> None:
        self.bobMass = 1     # kg
        self.rodLength = 2   # m
        self.gravity = 9.81  # m/s2

    def forward_dynamic(self, theta, thetaDot):
        return np.array([[thetaDot], [-self.gravity * np.sin(theta) / self.rodLength]])

    def simulate_euler(self):

        def dynamic(pState, input):
            return self.forward_dynamic(pState[0,0], pState[1,0])

        def control(pState, dState):
            return np.array([0]) # not used, system has no control

        def desired(pState, time):
            return np.array([0]) # not used, system has no control

        q0 = np.array([[np.pi / 10], [0]])  # initial state start at 45deg with 0 angular velo
        tSpan = (0, 10)
        dt = 0.0001 # dt must be so small so that rounding error is not effecting the dynamic of system making it get bigger and bigger
        intg = EulerNumericalIntegrator(dynamic, control, desired, q0, tSpan, dt)
        intg.simulation()
        timeSteps, states, desireds, controls = intg.simulation()

        # Plot the results
        plt.plot(timeSteps, states[0,:], label='$\\theta$(t) rad')
        plt.plot(timeSteps, states[1,:], label='$\omega$(t) rad/s')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    def simulate_gekko(self):
        m = GEKKO(remote=False)

        times = np.arange(0, 5, 0.001)
        m.time = times

        theta = m.Var(value=np.pi / 10, lb=-np.pi, ub=np.pi)
        thetaDot = m.Var(value=0)

        m.Equation(theta.dt() == thetaDot)
        m.Equation(thetaDot.dt() == -self.gravity * m.sin(theta) / self.rodLength)

        m.options.IMODE = 4
        m.solve(disp=True)

        plt.plot(m.time, theta.VALUE, label='$\\theta$(t) rad')
        plt.plot(m.time, thetaDot.VALUE, label='$\omega$(t) rad/s')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # gekko simulate
    pen = Pendulum()
    pen.simulate_gekko()

    # euler simulate
    pen.simulate_euler()