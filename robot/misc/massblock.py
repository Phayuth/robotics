import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from gekko import GEKKO
from simulator.integrator_euler import EulerNumericalIntegrator


class MassBlock:

    def __init__(self) -> None:
        self.mass = 2       # kg
        self.gravity = 9.81 # m/s2
        self.fricCoefSteelvSteel = 0.42
        self.fricForce = self.fricCoefSteelvSteel*self.mass*self.gravity

    def forward_dynamic(self, velocity, inputForce):
        return np.array([[velocity], [inputForce/self.mass]])

    def simulate_euler(self):
        def dynamic(pState, input):
            return self.forward_dynamic(pState[1,0], input)

        def control(pState, dState): # force input in newton
            if pState[0,0] < 4:
                return np.array([10])
            else:
                return np.array([0])

        def desired(pState, time):
            return np.array([0]) # not used

        q0 = np.array([[0], [0]])
        tSpan = (0, 10)
        dt = 0.01
        intg = EulerNumericalIntegrator(dynamic, control, desired, q0, tSpan, dt)
        timeSteps, states, desireds, controls = intg.simulation()

        # Plot the results
        plt.plot(timeSteps, states[0,:], label='distant m')
        plt.plot(timeSteps, states[1,:], label='velocity m/s')
        plt.plot(timeSteps, controls.flatten(), label='force newton')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    def simulate_gekko(self):
        m = GEKKO(remote=False)

        # times
        nt = 501
        times = np.linspace(0, 5, nt)
        m.time = times

        # state variable
        x = m.Var(value=0)
        v = m.Var(value=0)

        f = 10*np.ones(nt)
        force = m.Param(f)

        m.Equation(x.dt()==v)
        m.Equation(v.dt()==force/self.mass)

        m.options.IMODE = 4
        m.solve(disp=False)

        plt.plot(m.time, x.VALUE, label='distance m')
        plt.plot(m.time, v.VALUE, label='velocity m/s')
        plt.plot(m.time, force.VALUE, label='force newton')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()


if __name__=="__main__":
    # gekko simulate
    bm = MassBlock()
    bm.simulate_gekko()

    # euler simulate
    bm.simulate_euler()