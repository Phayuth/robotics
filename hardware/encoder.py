import numpy as np


def pulse_to_theta(gearBoxRatio, pulsePerRevo, pulseCount):
    return (2 * np.pi * pulseCount) / (gearBoxRatio * pulsePerRevo)


def theta_to_omega(thetaCurrent, thetaPrevious, omegaPrevious, cteFilter, timeStep):
    """
    Transfer function for determining the wheel velocity T(s)=s/(as+1) with some filter
    cteFilter = Some small constant for tuning signal and help filter out noise = 0.04 good
    """
    return (1 - timeStep / cteFilter) * omegaPrevious + (1 / cteFilter) * (thetaCurrent - thetaPrevious)