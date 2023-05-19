import numpy as np

a = np.array([[0,0,0],
             [1,1,1],
             [2,2,2]])

np.random.shuffle(a)
print(f"==>> b: \n{a}")