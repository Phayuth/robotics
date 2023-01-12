import numpy as np

# good old 2d rr
a1 = 1
a2 = 2
theta1 = 1
theta2 = 1

O0 = np.array([[0],
               [0],
               [0]])

O1 = np.array([[a1*np.cos(theta1)],
               [a1*np.sin(theta1)],
               [      0         ]])

O2 = np.array([[a1*np.cos(theta1) + a2*np.cos(theta1+theta2)],
               [a1*np.sin(theta1) + a2*np.sin(theta1+theta2)],
               [                  0                         ]])

Z0 = np.array([[0],
               [0],
               [1]])

Z1 = np.array([[0],
               [0],
               [1]])

# joint 1
# first transpose both row vector , then do cross product , the transpose to column vector back. 
# because of np.cross use row vector (I dont know how to use in properly yet)
Jv1 = np.transpose(np.cross(np.transpose(Z0),np.transpose(O2-O0))) 
Jw1 = Z0

# joint 2
Jv2 = np.transpose(np.cross(np.transpose(Z1),np.transpose(O2-O1)))
Jw2 = Z1

print(Jv1)
print(Jw1)

print(Jv2)
print(Jw2)

J1 = np.append(Jv1,Jw1,axis=0) # if not use axis = the result is 1x6, axis=0 the result is 6x1, axis=1 the result is 3x2
J2 = np.append(Jv2,Jw2,axis=0)

print(J1)
print(J2)

J = np.append(J1,J2,axis=1) # we want J = [J1 , J2] side by side => use axis = 1
print(J)