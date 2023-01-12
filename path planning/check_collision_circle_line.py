import numpy as np
# https://www.baeldung.com/cs/circle-line-segment-collision-detection

def trig_area(A,B,C):
    Ax = A[0,0]
    Ay = A[1,0]

    Bx = B[0,0]
    By = B[1,0]

    Cx = C[0,0]
    Cy = C[1,0]

    AB = np.array([[Bx - Ax],
                  [By - Ay]])

    AC = np.array([[Cx - Ax],
                  [Cy - Ay]])

    cross_prod = np.cross(np.transpose(AB),np.transpose(AC)) # cross prod is scalar AB[0,0]*AC[1,0] - AB[1,0]*AC[0,0]

    return abs(cross_prod)/2

def line_circle_collision(radius,O,P,Q):
    distPQ = P - Q
    print(distPQ)
    minimum_distance = 2*trig_area(O,P,Q)/np.linalg.norm(distPQ)
    print(minimum_distance)

    if minimum_distance <= radius:
        return True
    else:
        return False

# A = np.array([0,0])
# B = np.array([5,0])
# C = np.array([2.5,2.5])

# area = trig_area(A,B,C)
# print(area)

r = 1

O = np.array([[0],
             [0]])

P = np.array([[-6],
             [-6]])

Q = np.array([[-6],
              [6]])


collide = line_circle_collision(r,O,P,Q)
print(collide)