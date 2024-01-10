import matplotlib.pyplot as plt
import numpy as np

class MassSpringDamperSystem:
    def __init__(self, position, velocity, mass, damping_coefficient, spring_constant):
        self.position = position
        self.velocity = velocity
        self.mass = mass
        self.damping_coefficient = damping_coefficient
        self.spring_constant = spring_constant

    def update(self, target_position, time_step):
        acceleration = (target_position - self.position) * self.spring_constant / self.mass - self.velocity * self.damping_coefficient / self.mass
        self.velocity += acceleration * time_step
        self.position += self.velocity * time_step

# Initialize system parameters
initial_position = np.array([0.0, 0.0])
initial_velocity = np.array([0.0, 0.0])
mass = 1.0
damping_coefficient = 0.8
spring_constant = 1.0

system = MassSpringDamperSystem(initial_position, initial_velocity, mass, damping_coefficient, spring_constant)

# Matplotlib interactive plot setup
fig, ax = plt.subplots()
point, = ax.plot([], [], 'bo')

def update_point(event):
    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          (event.button, event.x, event.y, event.xdata, event.ydata))
    target_position = np.array([event.xdata, event.ydata])
    print(f"==>> target_position: \n{target_position}")
    system.update(target_position, 0.1)
    point.set_data(system.position[0], system.position[1])
    plt.draw()


cid = fig.canvas.mpl_connect('button_press_event', update_point)

# Set plot limits
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
plt.grid(True)

# Start the interactive plot
plt.show()