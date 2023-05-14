import numpy as np
import matplotlib.pyplot as plt


def cubic_coef(theta0, thetaF, tF):

    a0 = theta0
    a1 = 0
    a2 = (3*(thetaF - theta0))/(tF**2)
    a3 = (-2*(thetaF - theta0))/(tF**3)

    return a0, a1, a2, a3
    
def cubic_eq(a0, a1, a2, a3, t):

    theta = a0 + (a1*t) + (a2*(t**2)) + (a3*(t**3))
    thetaDot = a1 + 2*a2*t + 3*a3*(t**2)
    thetaDDot = 2*a2 + 6*a3*t

    return theta, thetaDot, thetaDDot

if __name__ == "__main__":
    # # theta
    # theta0 = -5
    # thetaF = 80
    # tF = 4

    # # coef
    # a0, a1, a2, a3 = cubic_coef(theta0, thetaF, tF)

    # # equation
    # t = np.arange(0,tF,0.01)
    # theta, thetaDot, thetaDDot = cubic_eq(a0, a1, a2, a3, t)


    # fig, (ax1, ax2, ax3) = plt.subplots(3)

    # ax1.set(xlabel = "time is second", ylabel = "joint pose in deg")
    # ax1.plot(t, theta)

    # ax2.set(xlabel = "time is second", ylabel = "joint vel in deg/s")
    # ax2.plot(t, thetaDot)

    # ax3.set(xlabel = "time is second", ylabel = "joint acc in deg/sec^2")
    # ax3.plot(t, thetaDDot)

    plt.show()

    # multi seg
    a00 = 5
    a10 = 0
    a20 = 15/2
    a30 = -5/2

    a01 = 15
    a11 = 0
    a21 = -75/4
    a31 = 25/4

    t0 = np.arange(0,2,0.01)
    t1 = np.arange(2,4,0.01)

    t = np.hstack((t0, t1))    
    print(f"==>> t.shape: \n{t.shape}")

    theta0, thetaDot0, thetaDDot0 = cubic_eq(a00, a10, a20, a30, t0)
    theta1, thetaDot1, thetaDDot1 = cubic_eq(a01, a11, a21, a31, t0)

    q = np.hstack((theta0, theta1))
    print(f"==>> q: \n{q}")
    dq = np.hstack((thetaDot0, thetaDot1))
    ddq = np.hstack((thetaDDot0, thetaDDot1))

    # Plotting Graph
    fig, (ax1, ax2, ax3) = plt.subplots(3)

    ax1.set(xlabel = "time is second", ylabel = "joint pose in deg")
    ax1.plot(t, q)

    ax2.set(xlabel = "time is second", ylabel = "joint vel in deg/s")
    ax2.plot(t, dq)

    ax3.set(xlabel = "time is second", ylabel = "joint acc in deg/sec^2")
    ax3.plot(t, ddq)

    plt.show()