import numpy as np
import matplotlib.pyplot as plt


obsxy = np.array([0.0, 0.0])
obsr = 0.5
br = 0.2  # robot radius
clearence = 0.1


def sdf(bxy, obsxy, obsr, br):
    dmin = np.linalg.norm(bxy - obsxy) - obsr - br
    return dmin


def cost_value(dmin, br, clearence):
    cost = max(clearence + br - dmin, 0)
    return cost


def click(event):
    if event.xdata is not None and event.ydata is not None:
        bxy = np.array([event.xdata, event.ydata])

        dmin = sdf(bxy, obsxy, obsr, br)
        cost = cost_value(dmin, br, clearence)

        bcircle = plt.Circle((bxy[0], bxy[1]), br, color="b", alpha=0.5)
        ax.add_artist(bcircle)

        clearencecircle = plt.Circle((bxy[0], bxy[1]), br + clearence, color="g", alpha=0.5)
        ax.add_artist(clearencecircle)

        ax.set_title(f"Clicked at: ({event.xdata:.2f}, {event.ydata:.2f}) | Cost: {cost:.2f}")
        ax.set_xlabel(f"Minimum distance: {dmin:.2f}")

        plt.draw()


fig, ax = plt.subplots()
obscircle = plt.Circle((obsxy[0], obsxy[1]), obsr, color="r", alpha=0.5)
ax.add_artist(obscircle)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect("equal")
fig.canvas.mpl_connect("button_press_event", click)
plt.show()
