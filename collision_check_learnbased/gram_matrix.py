import numpy as np

def is_lin_ind(G):
    det = np.linalg.det(G)
    if det != 0:
        print("It is linearly independent")
    else:
        print("It is not linearly independent")

V = np.array([[5,7],[1,9]])
VT = np.transpose(V)

P = np.array([[1,1],[2,2]])
PT = np.transpose(P)

G = VT @ V
GG = PT @ P

print(G)
is_lin_ind(G)
print(GG)
is_lin_ind(GG)

print(np.linalg.matrix_rank(G))
print(np.linalg.matrix_rank(GG))