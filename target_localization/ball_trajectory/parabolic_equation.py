import numpy as np

def parabolic(t, a, b, c, return_velo=False):
    pose = (a * (t**2)) + (b * t) + c
    if return_velo:
        velo = 2 * a * t + b
        return pose, velo
    else:
        return pose