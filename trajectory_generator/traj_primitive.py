import numpy as np


def circle(t):
    freq = 2 * np.pi / 30
    radius = 3

    x = radius * np.cos(freq * t)
    y = radius * np.sin(freq * t)

    xdot = -radius * freq * np.sin(freq * t)
    ydot = radius * freq * np.cos(freq * t)

    xddot = -radius * (freq**2) * np.cos(freq * t)
    yddot = -radius * (freq**2) * np.sin(freq * t)

    xdddot = radius * (freq**3) * np.sin(freq * t)
    ydddot = -radius * (freq**3) * np.cos(freq * t)

    vr = np.sqrt((xdot**2 + ydot**2))
    wr = ((xdot * yddot - ydot * xddot)) / ((xdot**2 + ydot**2))

    vdotr = (xdot * xddot + ydot * yddot) / vr
    wdotr = ((xdot * ydddot - ydot * xdddot) / (vr**2)) - ((2 * wr * vdotr) / vr)

    return x, y, vr, wr, ydot, xdot, vdotr, wdotr


def sin_at_45deg(t):
    freq = 2 * np.pi / 30
    radius = 5

    x = (np.cos(np.radians(45)) * t) - (np.sin(np.radians(45)) * radius * np.sin(freq * t))
    y = (np.sin(np.radians(45)) * t) + (np.cos(np.radians(45)) * radius * np.sin(freq * t))

    xdot = np.cos(np.radians(45)) - (np.sin(np.radians(45)) * radius * freq * np.cos(freq * t))
    ydot = np.sin(np.radians(45)) + (np.cos(np.radians(45)) * radius * freq * np.cos(freq * t))

    xddot = np.sin(np.radians(45)) * radius * (freq**2) * np.sin(freq * t)
    yddot = -np.cos(np.radians(45)) * radius * (freq**2) * np.sin(freq * t)

    xdddot = np.sin(np.radians(45)) * radius * (freq**3) * np.cos(freq * t)
    ydddot = -np.cos(np.radians(45)) * radius * (freq**3) * np.cos(freq * t)

    vr = np.sqrt((xdot**2 + ydot**2))
    wr = ((xdot * yddot - ydot * xddot)) / ((xdot**2 + ydot**2))

    vdotr = (xdot * xddot + ydot * yddot) / vr
    wdotr = ((xdot * ydddot - ydot * xdddot) / (vr**2)) - ((2 * wr * vdotr) / vr)

    return x, y, vr, wr, ydot, xdot, vdotr, wdotr


def fig_of_8(t):
    freq = 2 * np.pi / 30

    xRef = 1.1 + 0.7 * np.sin(freq * t)
    yRef = 0.9 + 0.7 * np.sin(2 * freq * t)

    dxRef = freq * 0.7 * np.cos(freq * t)
    dyRef = 2 * freq * 0.7 * np.cos(2 * freq * t)

    ddxRef = -(freq**2) * 0.7 * np.sin(freq * t)
    ddyRef = -4 * (freq**2) * 0.7 * np.sin(2 * freq * t)

    vRef = np.sqrt((dxRef**2) + (dyRef**2))
    wRef = ((dxRef * ddyRef) - (dyRef * ddxRef)) / ((dxRef**2) + (dyRef**2))

    return xRef, yRef, dxRef, dyRef, ddxRef, ddyRef, vRef, wRef
