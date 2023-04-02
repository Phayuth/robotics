import numpy as np
import collision_class

a = collision_class.obj_point3d(5,5,2)
b = collision_class.obj_point3d(5,5,5)
c = collision_class.obj_point2d(5,5)

trig = collision_class.obj_triangle([0,0],[10,0],[0,10])

collision = collision_class.intersect_point_v_point_3d(a,b)
collision = collision_class.intersect_triangle_v_point(trig,c)
r = 1

O = np.array([[0],
             [0]])

P = np.array([[-6],
             [-6]])

Q = np.array([[-6],
              [6]])


collide = collision_class.intersect_line_v_circle_collisio(r,O,P,Q)
print(collision)

rec1 = collision_class.obj_rec(0,0,5,5)
rec2 = collision_class.obj_rec(10,10,5,5)
collision = collision_class.intersect_rectangle_v_rectangle(rec1,rec2)
print(collision)