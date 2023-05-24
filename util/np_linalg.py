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



# shuffle dataset
a = np.array([[0,0,0],
             [1,1,1],
             [2,2,2]])

np.random.shuffle(a)
print(f"==>> b: \n{a}")


def is_linear_independent(G):
    det = np.linalg.det(G)
    if det != 0:
        print("It is linearly independent")
    else:
        print("It is not linearly independent")


V = np.array([[5, 7], [1, 9]])
VT = np.transpose(V)

P = np.array([[1, 1], [2, 2]])
PT = np.transpose(P)

G = V @ VT
GG = P @ PT

print(G)
is_linear_independent(G)

print(GG)
is_linear_independent(GG)

print(np.linalg.matrix_rank(G))
print(np.linalg.matrix_rank(GG))