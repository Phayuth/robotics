import numpy as np
import collision_class

a = collision_class.point3d_obj(5,5,2)
b = collision_class.point3d_obj(5,5,5)
c = collision_class.point2d_obj(5,5)

trig = collision_class.triangle_obj([0,0],[10,0],[0,10])

collision = collision_class.point_to_point_3d(a,b)
collision = collision_class.triangle_to_point(trig,c)
r = 1

O = np.array([[0],
             [0]])

P = np.array([[-6],
             [-6]])

Q = np.array([[-6],
              [6]])


collide = collision_class.line_circle_collision(r,O,P,Q)
print(collision)

rec1 = collision_class.sqr_rec_2d_obj(0,0,5,5)
rec2 = collision_class.sqr_rec_2d_obj(10,10,5,5)
collision = collision_class.intersect_rectangle_v_rectangle(rec1,rec2)
print(collision)