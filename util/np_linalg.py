import numpy as np

a = np.array([0,0,1]).reshape(3,1)
b = np.array([1,0,0]).reshape(3,1)

c = np.cross(a,b,axisa=0, axisb=0, axisc=0)
print(f"==>> c.shape: \n{c.shape}")
print(f"==>> c: \n{c}")


c = np.cross(a,b,axis=0)
print(f"==>> c: \n{c}")

d = 2 @ a 
print(f"==>> d: \n{d}")