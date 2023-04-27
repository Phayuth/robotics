import numpy as np
import collision_class
import matplotlib.pyplot as plt

a = collision_class.ObjPoint3D(5, 5, 2)
b = collision_class.ObjPoint3D(5, 5, 5)
c = collision_class.ObjPoint2D(5, 5)

trig = collision_class.ObjTriangle([0, 0], [10, 0], [0, 10])

collision = collision_class.intersect_point_v_point_3d(a, b)
collision = collision_class.intersect_triangle_v_point(trig, c)
r = 1

O = np.array([[0], [0]])

P = np.array([[-6], [-6]])

Q = np.array([[-6], [6]])

collide = collision_class.intersect_line_v_circle(r, O, P, Q)
print(collision)

rec1 = collision_class.ObjRec(0, 0, 5, 5)
rec2 = collision_class.ObjRec(10, 10, 5, 5)
collision = collision_class.intersect_rectangle_v_rectangle(rec1, rec2)
print(collision)


# recWithAngle = collision_class.ObjRec(1,1,1,5,angle=2)
# line = collision_class.ObjLine2D(-1,1,0,5)
# collide = collision_class.intersect_line_v_rectangle(line, recWithAngle)
# print(f"==>> collide: \n{collide}")
# line.plot()
# recWithAngle.plot()
# plt.show()



from util.coord_transform import polar2cats, circle_plt


plt.axes().set_aspect('equal')
plt.axvline(x=0, c="black")
plt.axhline(y=0, c="black")
xTarg = 2
yTarg = 2
alpha = 2

hD = 0.5
wD = 2
rCrop = 0.5




xTopStart = rCrop*np.cos(alpha - np.pi/2) + xTarg
yTopStart = rCrop*np.sin(alpha - np.pi/2) + yTarg

xBotStart = (rCrop)*np.cos(alpha + np.pi/2) + xTarg
yBotStart = (rCrop)*np.sin(alpha + np.pi/2) + yTarg


# recTop = collision_class.ObjRec(xTopStart, yTopStart, hD, wD, angle=alpha)
recBot = collision_class.ObjRec(xBotStart, yBotStart, hD, wD, angle=alpha)
circle_plt(xTarg, yTarg, radius=rCrop)
# recTop.plot()
recBot.plot()
plt.show()