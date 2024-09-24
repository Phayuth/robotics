from gekko import GEKKO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


qsx = -5
qsy = 5
qex = 5
qey = 5

m = GEKKO()
q = m.Array(m.Var, 8)

# minimize square cost
m.Obj(((qsx - q[0]) ** 2 + (qsy - q[1]) ** 2) + ((q[0] - q[2]) ** 2 + (q[1] - q[3]) ** 2) + ((q[2] - q[4]) ** 2 + (q[3] - q[5]) ** 2) + ((q[4] - q[6]) ** 2 + (q[5] - q[7]) ** 2) + ((q[6] - qex) ** 2 + (q[7] - qey) ** 2))

m.Equations(
    [
        # convex feasible set
        q[0] >= -3,
        q[0] <= 0,
        q[2] >= -3,
        q[2] <= 0,
        q[1] >= 0,
        q[1] <= 3,
        q[3] >= 0,
        q[3] <= 3,
        q[4] >= 0,
        q[4] <= 3,
        q[6] >= 0,
        q[6] <= 3,
        q[5] <= 3,
        q[5] >= 0,
        q[7] <= 3,
        q[7] >= 0,
        # continuity
        q[2] == q[4],
        q[3] == q[5],
    ]
)
m.solve(disp=False)


qopt = [q[i].value[0] for i in range(8)]
qopt = np.array(qopt).reshape(-1, 2)

plt.plot(qopt[:, 0], qopt[:, 1], color="red", marker="*", markerfacecolor="green")
plt.plot([qsx, qex], [qsy, qey], "b^")
plt.plot([qsx] + qopt[:, 0].tolist() + [qex], [qsy] + qopt[:, 1].tolist() + [qey])
plt.gca().set_aspect("equal", "box")
plt.gca().add_patch(Rectangle((-3, 0), 3, 3, edgecolor="green", facecolor="none"))
plt.gca().add_patch(Rectangle((0, 0), 3, 3, edgecolor="yellow", facecolor="none"))
plt.axhline()
plt.axvline()
plt.show()
