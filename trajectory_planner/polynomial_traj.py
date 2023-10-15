import numpy as np
import matplotlib.pyplot as plt


class PolynomialEquation:

    def cubic_polynomial(a0, a1, a2, a3, t):
        position = a0 + (a1*t) + (a2 * (t**2)) + (a3 * (t**3))
        firstDeriv = a1 + 2*a2*t + 3 * a3 * (t**2)
        secondDeriv = 2*a2 + 6*a3*t
        return position, firstDeriv, secondDeriv

    def cubic_coef(posStart, posFinal, tF):
        a0 = posStart
        a1 = 0
        a2 = (3 * (posFinal-posStart)) / (tF**2)
        a3 = (-2 * (posFinal-posStart)) / (tF**3)
        return a0, a1, a2, a3

    def cubic_coef_specify_velo(posStart, veloStart, posFinal, veloFinal, tF):
        a0 = posStart
        a1 = veloStart
        a2 = (3 * (posFinal-posStart)) / (tF**2) - (2*veloStart) / tF - veloFinal/tF
        a3 = (-2 * (posFinal-posStart)) / (tF**3) + (veloFinal+veloStart) / (tF**2)
        return a0, a1, a2, a3

    def quintic_polynomial(a0, a1, a2, a3, a4, a5, t):
        position = a0 + (a1*t) + (a2 * (t**2)) + (a3 * (t**3)) + (a4 * (t**4)) + (a5 * (t**5))
        firstDeriv = a1 + 2*a2*t + 3 * a3 * (t**2) + 4 * a4 * (t**3) + 5 * a5 * (t**4)
        secondDeriv = 2*a2 + 6*a3*t + 12 * a4 * (t**2) + 20 * a5 * (t**3)
        return position, firstDeriv, secondDeriv

    def quintic_coef(posStart, veloStart, accStart, posFinal, veloFinal, accFinal, tF):
        a0 = posStart
        a1 = veloStart
        a2 = accStart / 2
        a3 = ((20*posFinal) - (20*posStart) - ((12*veloStart + 8*veloFinal) * tF) - ((3*accStart - accFinal) * (tF**2))) / (2 * (tF**3))
        a4 = (-(30 * posFinal) + (30*posStart) + ((16*veloStart + 14*veloFinal) * tF) + ((3*accStart - 2*accFinal) * (tF**2))) / (2 * (tF**4))
        a5 = ((12*posFinal) - (12*posStart) - ((6*veloStart + 6*veloFinal) * tF) - ((accStart-accFinal) * (tF**2))) / (2 * (tF**5))
        return a0, a1, a2, a3, a4, a5

    def plot_trajectory(pos, velo, acc, t, ax1, ax2, ax3):
        ax1.set(ylabel="joint pose in deg")
        ax1.plot(t, pos)
        ax2.set(ylabel="joint vel in deg/s")
        ax2.plot(t, velo)
        ax3.set(xlabel="time is second", ylabel="joint acc in deg/sec^2")
        ax3.plot(t, acc)


if __name__ == "__main__":
    # cubic
    tF = 4
    t = np.arange(0, tF, 0.01)
    # cubicCoeff = PolynomialEquation.cubic_coef(posStart=-5, posFinal=80, tF=tF)
    cubicCoeff = PolynomialEquation.cubic_coef_specify_velo(posStart=0, veloStart=0.5, posFinal=5, veloFinal=0, tF=tF)
    q, dq, ddq = PolynomialEquation.cubic_polynomial(*cubicCoeff, t)

    # cubic multi seg
    t0 = np.arange(0, 2, 0.01)
    t1 = np.arange(2, 4, 0.01)
    t = np.hstack((t0, t1))
    theta0, thetaDot0, thetaDDot0 = PolynomialEquation.cubic_polynomial(a0=5, a1=0, a2=15 / 2, a3=-5 / 2, t=t0)
    theta1, thetaDot1, thetaDDot1 = PolynomialEquation.cubic_polynomial(a0=15, a1=0, a2=-75 / 4, a3=25 / 4, t=t0)
    q = np.hstack((theta0, theta1))
    dq = np.hstack((thetaDot0, thetaDot1))
    ddq = np.hstack((thetaDDot0, thetaDDot1))

    # quintic
    # tF = 5
    # t = np.arange(0, tF, 0.01)
    # quinticCoeff = PolynomialEquation.quintic_coef(posStart=0, veloStart=0, accStart=0, posFinal=5, veloFinal=0, accFinal=0, tF=tF)
    # q, dq, ddq = PolynomialEquation.quintic_polynomial(*quinticCoeff, t)

    # plot
    fig, axes = plt.subplots(3)
    PolynomialEquation.plot_trajectory(q, dq, ddq, t, *axes)
    plt.show()
