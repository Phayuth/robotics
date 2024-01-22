import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from gekko import GEKKO
from simulator.integrator_euler import EulerNumericalIntegrator


class DCMotor:

    def __init__(self):
        # dcmotor lumped constant
        self.a = 28.24
        self.b = 53.12
        self.c = 24.5

    def forward_dynamic(self, omega, voltage):
        return -self.a * omega + self.b * voltage - self.c * np.sign(omega)

    def simulate_euler(self):

        def dynamic(pState, input):
            return self.forward_dynamic(pState, input)

        def control(pState, dState):  # dState here is just time
            vol = 14.0
            freq = np.pi / 2
            return vol * np.sin(dState * freq)

        def desired(pState, time):
            return np.array([time])  # give time

        q0 = np.array([0.0])
        tSpan = (0, 10)
        dt = 0.0001  # increase dt to see chattering effect
        intg = EulerNumericalIntegrator(dynamic, control, desired, q0, tSpan, dt)
        intg.simulation()
        timeSteps, states, desireds, controls = intg.simulation()

        # Plot the results
        plt.plot(timeSteps, states.flatten(), label='$\omega$(t) rad/s')
        plt.plot(timeSteps, controls.flatten(), label='inVoltage(t) v')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    def simulate_gekko(self):
        m = GEKKO(remote=False)

        # Time points
        nt = 101
        m.time = np.linspace(0, 10, nt)

        # Define variables
        omega = m.Var(value=0.0)
        voltage = np.zeros(nt)
        voltage[10:] = 24.0  # step input at t=1sec
        voltage[50:] = 16.0  # step input at t=5sec
        inVoltage = m.Param(value=voltage)

        m.Equation(omega.dt() == -self.a * omega + self.b * inVoltage)
        m.fix(omega, pos=0, val=0.0)
        m.options.IMODE = 4  # Set up the solver options Dynamic simulation
        m.solve(disp=False)

        plt.plot(m.time, omega.value, label='$\omega$(t) rad/s')
        plt.plot(m.time, inVoltage.value, label='inVoltage(t) v')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # gekko simulate
    motor = DCMotor()
    motor.simulate_gekko()

    # euler simulation
    motor.simulate_euler()