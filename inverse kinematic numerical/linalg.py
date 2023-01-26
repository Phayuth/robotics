import numpy as np


J = np.array([[25,7],[88,5]])
e = np.array([[7],[2]])
U, D, VT = np.linalg.svd(J)
V = np.transpose(VT)


v1 = np.array([[V[0,0]],[V[1,0]]])
v2 = np.array([[V[0,1]],[V[1,1]]])

u1 = np.array([[U[0,0]],[U[1,0]]])
u2 = np.array([[U[0,1]],[U[1,1]]])

alpha1 = np.dot(np.transpose(e),u1)
alpha2 = np.dot(np.transpose(e),u2)

tau1 = 0.2
tau2 = 0.5

delta_theta=alpha1*tau1*v1 + alpha2*tau2*v2
# print(delta_theta)


# print(U[0,0])
# print(U[1,0])
# u1 = np.array([[U[0,0]],[U[1,0]]])
# alpha_method1 = np.dot(np.transpose(e),u1)
# print(alpha_method1)
# alpha_method2 = np.transpose(u1) @ e
# print(alpha_method2)

def clampMagAbs(w,d): 
    if np.max(abs(w))<=d:
        return w
    else:
        return d*(w/(np.max(abs(w))))

def rho11(l1,l2,theta1,theta2):
    eq1 = -l1*np.sin(theta1)
    eq2 =  l1*np.cos(theta1)
    vector = np.array([[eq1],[eq2]])
    return np.linalg.norm(vector)

def rho12(l1,l2,theta1,theta2):
    eq1 = 0
    eq2 = 0
    vector = np.array([[eq1],[eq2]])
    return np.linalg.norm(vector)

def rho21(l1,l2,theta1,theta2):
    eq1 = -l1*np.sin(theta1) - l2*np.sin(theta1+theta2)
    eq2 =  l1*np.cos(theta1) + l2*np.cos(theta1+theta2)
    vector = np.array([[eq1],[eq2]])
    return np.linalg.norm(vector)

def rho22(l1,l2,theta1,theta2):
    eq1 = -l2*np.sin(theta1+theta2)
    eq2 =  l2*np.cos(theta1+theta2)
    vector = np.array([[eq1],[eq2]])
    return np.linalg.norm(vector)

l1 = 1
l2 = 1
theta1 = 1
theta2 = 1

v11 = V[0,0]
v12 = V[0,1]
v21 = V[1,0]
v22 = V[1,1]

sigma1 = D[0]
sigma2 = D[1]

M11 = (1/sigma1)*(np.abs(v11)*rho11(l1,l2,theta1,theta2) + np.abs(v21)*rho12(l1,l2,theta1,theta2))
M12 = (1/sigma1)*(np.abs(v11)*rho21(l1,l2,theta1,theta2) + np.abs(v21)*rho22(l1,l2,theta1,theta2))

M21 = (1/sigma2)*(np.abs(v12)*rho11(l1,l2,theta1,theta2) + np.abs(v22)*rho12(l1,l2,theta1,theta2))
M22 = (1/sigma2)*(np.abs(v12)*rho21(l1,l2,theta1,theta2) + np.abs(v22)*rho22(l1,l2,theta1,theta2))

M1 = M11 + M12
M2 = M21 + M22

N1 = np.sum(np.abs(u1))
N2 = np.sum(np.abs(u2))

gama_max = np.pi/4
gama1 = np.minimum(1,N1/M1)*gama_max
gama2 = np.minimum(1,N2/M2)*gama_max

# print(gama1)
# print(gama2)

c1 = (1/sigma1) * alpha1 * v1
c2 = (1/sigma2) * alpha2 * v2

phi1 = clampMagAbs(c1,gama1)
phi2 = clampMagAbs(c2,gama2)

print(phi1)
print(phi2)

deltheta = clampMagAbs(phi1+phi2,gama_max)
print(deltheta)