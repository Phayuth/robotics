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

    def cubic_via_point():
        def cubic_3_via_point(theta_0,theta_v,theta_g,t,tf):
            # case tf = tf1 = tf2
            a10 = theta_0
            a11 = 0
            a12 = (12*theta_v - 3*theta_g - 9*theta_0)/(4*(tf**2))
            a13 = (-8*theta_v + 3*theta_g + 5*theta_0)/(4*(tf**3))

            a20 = theta_v
            a21 = (3*theta_g - 3*theta_0)/(4*tf)
            a22 = (-12*theta_v + 6*theta_g + 6*theta_0)/(4*(tf**2))
            a23 = (8*theta_v - 5*theta_g - 3*theta_0)/(4*(tf**3))

            seg1_theta = a10 + (a11*t) + (a12*(t**2)) + (a13*(t**3))
            seg1_theta_dot = a11 + 2*a12*t + 3*a13*(t**2)
            seg1_theta_ddot = 2*a12 + 6*a13*t

            seg2_theta = a20 + (a21*t) + (a22*(t**2)) + (a23*(t**3))
            seg2_theta_dot = a21 + 2*a22*t + 3*a23*(t**2)
            seg2_theta_ddot = 2*a22 + 6*a23*t

            return seg1_theta,seg1_theta_dot,seg1_theta_ddot,seg2_theta,seg2_theta_dot,seg2_theta_ddot

        theta_0 = 0
        theta_v = 5
        theta_g = 10
        t_f = 10
        t = np.arange(0,t_f,0.01)
        seg1_theta,seg1_theta_dot,seg1_theta_ddot,seg2_theta,seg2_theta_dot,seg2_theta_ddot = cubic_3_via_point(theta_0,theta_v,theta_g,t,t_f)
        tt = np.arange(0,2*t_f, 0.01)
        plt.plot(tt,np.hstack((seg1_theta, seg2_theta)))
        plt.show()

    def linear_parabolic_blend():
        q0=-5; qf = 80; tf = 4
        min_acc = 4*abs(qf-q0)/tf**2
        print("acc needs to be bigger than ", min_acc)
        mode_aorv = 0
        if mode_aorv == 0:
            acc = 30
            tb = tf/2 -np.sqrt( acc**2 * tf**2 -4*acc*np.abs(qf-q0)) /2/acc
            if (qf-q0)>=0:
                vel = acc * tb
            else:
                acc = acc * -1
                vel = acc * tb
        else:
            vel = 25
            tb = (vel*tf - abs(qf-q0))/vel
            if (qf-q0)>=0:
                acc = vel / tb
            else:
                vel = vel * -1
                acc = vel/tb

        # Acceleration Region  # 1st Region
        t1 = np.arange(0, tb, step=0.01)
        q1 = q0 + vel/(2*tb) * (t1**2)
        dq1 = acc * t1
        ddq1 = acc*np.ones(np.size(t1))

        # Constant Velocity Region # 2nd Region
        t2   = np.arange(tb, tf-tb, step=0.01)
        q2  = (qf+q0-vel*tf)/2 + vel* t2
        dq2 = vel*np.ones(np.size(t2))
        ddq2 = 0*np.ones(np.size(t2))

        # Decceleration Region # 3rd Region
        t3 = np.arange(tf-tb, tf+0.01, step=0.01)
        q3 = qf - acc/2 * tf**2 + acc*tf*t3 - acc/2*t3**2
        dq3 = acc * tf - acc*t3
        ddq3 = -acc * np.ones(np.size(t3))

        # Total Region
        t = np.concatenate((t1,t2,t3))
        q = np.concatenate((q1,q2,q3))
        dq = np.concatenate((dq1,dq2,dq3))
        ddq = np.concatenate((ddq1,ddq2,ddq3))

        # Plotting Graph
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        ax1.plot(t, q, ylabel = "joint pose in deg")
        ax2.plot(t, dq, ylabel = "joint vel in deg/s")
        ax3.plot(t, ddq, ylabel = "joint acc in deg/sec^2")
        plt.legend()
        plt.show()


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
