import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import geometry.geometry_class as geometry_class
import matplotlib.pyplot as plt
from robot.planar_rrr import PlanarRRR

# a = geometry_class.ObjPoint3D(5, 5, 2)
# b = geometry_class.ObjPoint3D(5, 5, 5)
# c = geometry_class.ObjPoint2D(5, 5)

# trig = geometry_class.ObjTriangle([0, 0], [10, 0], [0, 10])

# collision = geometry_class.intersect_point_v_point_3d(a, b)
# collision = geometry_class.intersect_triangle_v_point(trig, c)
# r = 1

# O = np.array([[0], [0]])

# P = np.array([[-6], [-6]])

# Q = np.array([[-6], [6]])

# collide = geometry_class.intersect_line_v_circle(r, O, P, Q)
# print(collision)

# rec1 = geometry_class.ObjRec(0, 0, 5, 5)
# rec2 = geometry_class.ObjRec(10, 10, 5, 5)
# collision = geometry_class.intersect_rectangle_v_rectangle(rec1, rec2)
# print(collision)


# recWithAngle = geometry_class.ObjRec(1,1,1,5,angle=2)
# line = geometry_class.ObjLine2D(-1,1,0,5)
# collide = geometry_class.intersect_line_v_rectangle(line, recWithAngle)
# print(f"==>> collide: \n{collide}")
# line.plot()
# recWithAngle.plot()
# plt.show()



robot = PlanarRRR()
theta = np.array([0.7,0,0]).reshape(3,1)

rec1 = geometry_class.ObjRectangle(x=1.5, y=1.25, h=0.2, w=1, angle=1)
rec2 = geometry_class.ObjRectangle(x=1.5, y=0.5, h=0.2, w=1, angle=0.3)

linkPose = robot.forward_kinematic(theta, return_link_pos=True)
linearm1 = geometry_class.ObjLine2D(linkPose[0][0], linkPose[0][1], linkPose[1][0], linkPose[1][1])
linearm2 = geometry_class.ObjLine2D(linkPose[1][0], linkPose[1][1], linkPose[2][0], linkPose[2][1])
linearm3 = geometry_class.ObjLine2D(linkPose[2][0], linkPose[2][1], linkPose[3][0], linkPose[3][1])

obsList = [rec1, rec2]
for i in obsList:
    col1 = geometry_class.intersect_line_v_rectangle(linearm1, i)
    print(f"==>> col1: \n{col1}")
    col2 = geometry_class.intersect_line_v_rectangle(linearm2, i)
    print(f"==>> col2: \n{col2}")
    col3 = geometry_class.intersect_line_v_rectangle(linearm3, i)
    print(f"==>> col3: \n{col3}")

robot.plot_arm(theta, plt_basis=True)
for obs in obsList:
    obs.plot()
plt.show()




# DONT DELETE, Ellipse Check ---------------------------------------------------------------------------------------
def if_point_in_hyperellipsoid(point, ellipsoidRM, ellipsoidAxis, ellipsoidCenter):
    point = point.reshape(2,1)
    ellipsoidCenter = ellipsoidCenter.reshape(2, 1)
    ellipsoidAxis = ellipsoidAxis.reshape(2, 1)

    pointCheck = point - ellipsoidCenter
    pointCheckRotateBack = ellipsoidRM.T @ pointCheck
    mid = pointCheckRotateBack / ellipsoidAxis
    midsq = mid**2
    eq = sum(midsq)
    if eq <= 1.0:
        return True
    else:
        return False

def plot_ellipse(center, axis, rotation_matrix):
    # Generate points along the local x-axis
    t = np.linspace(0, 2*np.pi, 100)
    x_local = axis[0] * np.cos(t)
    y_local = axis[1] * np.sin(t)

    # Combine x and y coordinates to form points
    local_points = np.vstack((x_local, y_local))

    # Apply rotation to the points using the given rotation matrix
    rotated_points = rotation_matrix @ local_points

    # Translate the rotated points to the ellipse's center
    translated_points = rotated_points + center.reshape(-1, 1)

    # Plot the rotated and translated ellipse
    plt.figure(figsize=(6, 6))
    plt.plot(translated_points[0], translated_points[1], label='Rotated and Translated Ellipse')
    plt.scatter(center[0], center[1], color='red', label='Center')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Ellipse with Rotation and Translation')
    plt.legend()
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')


# Example usage
center = np.array([0, 4])
axis = np.array([5, 3])
theta = 1.0
rotation_matrix_og = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])

pointToCheck1 = np.array([4.58, 1.14])
pointToCheck2 = np.array([0.0, 0.0])
pointToCheck3 = np.array([4.0, 3.8])
pointToCheck4 = np.array([-2.5, 6.4])

state1 = if_point_in_hyperellipsoid(pointToCheck1, rotation_matrix_og, axis, center)
print(f"==>> state1: \n{state1}")
print("-----------------------------------------------------------------")

state2 = if_point_in_hyperellipsoid(pointToCheck2, rotation_matrix_og, axis, center)
print(f"==>> state2: \n{state2}")
print("-----------------------------------------------------------------")

state3 = if_point_in_hyperellipsoid(pointToCheck3, rotation_matrix_og, axis, center)
print(f"==>> state3: \n{state3}")
print("-----------------------------------------------------------------")

state4 = if_point_in_hyperellipsoid(pointToCheck4, rotation_matrix_og, axis, center)
print(f"==>> state4: \n{state4}")
print("-----------------------------------------------------------------")

# plot_ellipse(center, axis, rotation_matrix)
plot_ellipse(center, axis, rotation_matrix_og)

plt.scatter(pointToCheck1[0], pointToCheck1[1], color='red')
plt.scatter(pointToCheck2[0], pointToCheck2[1], color='red')
plt.scatter(pointToCheck3[0], pointToCheck3[1], color='red')
plt.scatter(pointToCheck4[0], pointToCheck4[1], color='red')

plt.show()




# superellipsoid 
s = geometry_class.ObjSuperEllipsoid2D(a=10, b=10, n=4)
s.plot()
plt.show()