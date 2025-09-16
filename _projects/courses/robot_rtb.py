import roboticstoolbox as rtb
import numpy as np
from spatialmath import SE3

robot = rtb.models.DH.Planar2()
robot.a1 = 1
robot.a2 = 1


def run_fk():
    theta = [0, 0]
    x = robot.fkine(theta)
    print(x)


def run_ik():
    x = SE3.Trans(1.0, 1.0, 0.0)
    q = robot.ik_LM(x)
    q = q[0]
    print(q)


def run_jac():
    theta = [0, 0]

    # Jacobian in world frame
    J = robot.jacob0(theta)
    print(J)

    # Jacobian Analytical in world frame
    Ja = robot.jacob0_analytical(theta)
    print(Ja)

    # Jacobian in end-effector frame
    Je = robot.jacobe(theta)
    print(Je)


def run_animation():
    q0 = [0, 0]
    qe = [np.pi / 2, np.pi / 2]
    traj = rtb.jtraj(q0, qe, 50)
    robot.plot(traj.q, backend="pyplot", loop=True, shadow=False)


if __name__ == "__main__":
    run_animation()
