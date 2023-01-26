import numpy as np



pointa = [0,0]
pointb = [0,0]

dist = np.array(pointa) - np.array(pointb)
d = np.linalg.norm(dist)

if d == 0:
    print("collision")
else:
    print("No collision")