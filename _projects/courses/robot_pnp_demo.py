from spatialmath import SE3
import numpy as np
import roboticstoolbox as rtb

robot = rtb.models.Panda()

# determine configuration
Tpick = SE3.Trans(0.6, -0.3, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
sol = robot.ik_LM(Tpick)
q_pick = sol[0]

Tplace = SE3.Trans(0.6, 0.3, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
sol2 = robot.ik_LM(Tplace)
q_release = sol2[0]

# compose a path
n = 50
q_prepick = robot.qr
path_pp_to_p = rtb.jtraj(q_prepick, q_pick, n)
path_p_to_pp = np.flip(path_pp_to_p.q, axis=0)
path_pp_to_r = rtb.jtraj(q_prepick, q_release, n)
path_r_to_pp = np.flip(path_pp_to_r.q, axis=0)
path = np.vstack((path_pp_to_p.q, path_p_to_pp, path_pp_to_r.q, path_r_to_pp))

# play
robot.plot(path, backend="swift", loop=True)
