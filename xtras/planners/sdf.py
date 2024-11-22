import numpy as np

c1 = np.array([2, 0, 1.5])  # x y r
c2 = np.array([5, 0, 1.5])

ec = c1 - c2
ctcd = np.linalg.norm(ec[..., 0:2])
print(f"> ctcd: {ctcd}")

rs = c1 + c2
rsum = rs[..., -1]
print(f"> rsum: {rsum}")


sdf = ctcd - rsum
print(f"> sdf: {sdf}")


X = np.linspace(-10, 10, num=1000)
Y = np.linspace(-10, 10, num=1000)

f = np.empty(shape=(1000, 1000))
for ix, ex in enumerate(X):
    for iy, ey in enumerate(Y):
        c2 = np.array([ex, ey, 1.5])
        ec = c1 - c2
        ctcd = np.linalg.norm(ec[..., 0:2])
        rs = c1 + c2
        rsum = rs[..., -1]
        sdf = ctcd - rsum
        f[ix, iy] = sdf
print(f"> f: {f}")


import matplotlib.pyplot as plt

plt.imshow(f, cmap="Greys")
plt.colorbar()
plt.show()


plt.contourf(f)
plt.show()

a = np.random.randint(low=0, high=10, size=(10,10))
print(f"> a: \n{a}")

d = 2
# for ki in range(0, 10, d):
#     print(a[..., ki:ki*d])


for i in range(0, a.shape[1], d):
    print(f"Columns {i} and {i+1}:\n", a[:, i:i+d], "\n")