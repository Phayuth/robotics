import numpy as np

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

    cross = np.cross(np.transpose(AB),np.transpose(AC))

    return abs(cross)/2

def distance(A,B):
    d = A - B
    d = np.linalg.norm(d)
    return d


def line_seg_circle_col(radius,O,P,Q):
    # min_dist = float('inf')
    max_dist = np.amax(np.array([distance(O,P),distance(O,Q)]))

    dot_OPQP = np.transpose(O-P) @ (Q-P)
    dot_OQPQ = np.transpose(O-Q) @ (P-Q)

    if dot_OPQP > 0 and dot_OQPQ > 0:
        min_dist = 2*trig_area(O,P,Q)/distance(P,Q)
        
    else:
        min_dist = np.amin(np.array([distance(O,P),distance(O,Q)]))

    print(min_dist)
    print(max_dist)

    if min_dist <= radius and max_dist >= radius:
        return True
    
    else:
        return False


r = 2
O = np.array([[0],[0]])
P = np.array([[-3],[-3]])
Q = np.array([[0],[0]])

cl = line_seg_circle_col(r,O,P,Q)
print(cl)