import numpy as np

def clampMag(w,d):
    if np.linalg.norm(w)<=d:
        return w
    else:
        return d*(w/np.linalg.norm(w))

def clampMagAbs(w,d): 
    if np.max(abs(w))<=d:
        return w
    else:
        return d*(w/(np.max(abs(w))))