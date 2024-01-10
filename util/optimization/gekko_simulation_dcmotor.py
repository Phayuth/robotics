from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

# dcmotor lumped constan
a = 28.24
b = 53.12
c = 24.5

# Create a Gekko model
m = GEKKO(remote=False)

# Time points
nt = 101
m.time = np.linspace(0, 10, nt)

# Define variables
omega = m.Var(value=0.0)
inVoltage = m.Param(value=np.zeros(nt))
inVoltage[10:] = 24.0  # Step input at t=1 until the end [in V]

# Define the differential equation
m.Equation(omega.dt() == -a * omega + b * inVoltage)

# Set the endpoint conditions
m.fix(omega, pos=0, val=0.0)

# Set up the solver options
m.options.IMODE = 4  # Dynamic simulation

# Solve the optimization problem
m.solve(disp=False)

# Plot the results
plt.plot(m.time, omega.value, label='$\omega$(t) rad/s')
plt.plot(m.time, inVoltage.value, label='inVoltage(t) v')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
