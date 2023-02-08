import numpy as np

class ur5e:
    def __init__(self):
        # DH parameter is from this publication
        # https://www.researchgate.net/publication/347021253_Mathematical_Modelling_and_Simulation_of_Human-Robot_Collaboration
        self.a     = np.array([      [0], [-0.425], [-0.3922],       [0],        [0],      [0]])
        self.alpha = np.array([[np.pi/2],      [0],       [0], [np.pi/2], [-np.pi/2],      [0]])
        self.d     = np.array([ [0.1625],      [0],       [0],  [0.1333],   [0.0997], [0.0996]])

    def dh_transformation(self,theta,alpha,d,a):
        R = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                      [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                      [            0,                np.sin(alpha),                np.cos(alpha),               d],
                      [            0,                            0,                            0,               1]])
        return R

    def forward_kinematic(self,theta):

        A1 = self.dh_transformation(theta[0,0],self.alpha[0,0],self.d[0,0],self.a[0,0])
        A2 = self.dh_transformation(theta[1,0],self.alpha[1,0],self.d[1,0],self.a[1,0])
        A3 = self.dh_transformation(theta[2,0],self.alpha[2,0],self.d[2,0],self.a[2,0])
        A4 = self.dh_transformation(theta[3,0],self.alpha[3,0],self.d[3,0],self.a[3,0])
        A5 = self.dh_transformation(theta[4,0],self.alpha[4,0],self.d[4,0],self.a[4,0])
        A6 = self.dh_transformation(theta[5,0],self.alpha[5,0],self.d[5,0],self.a[5,0])

        # T01 = A1
        # T02 = A1 @ A2
        # T03 = A1 @ A2 @ A3
        # T04 = A1 @ A2 @ A3 @ A4
        # T05 = A1 @ A2 @ A3 @ A4 @ A5
        T06 = A1 @ A2 @ A3 @ A4 @ A5 @ A6

        # pe = np.array([[0],[0],[0],[1]]) # the fourth element MUST be equal to 1
        # p01 = T01 @ pe
        # p02 = T02 @ pe
        # p03 = T03 @ pe
        # p04 = T04 @ pe
        # p05 = T05 @ pe
        # p06 = T06 @ pe


        # https://www.daslhub.org/unlv/courses/me729-sp/week03/lecture/Note_02_Forward_Kinematics.pdf
        # https://robotics.stackexchange.com/questions/8516/getting-pitch-yaw-and-roll-from-rotation-matrix-in-dh-parameter
        # x - y - z sequence
        # tan(roll) = r32/r33
        # tan(pitch)= -r31/(sqrt(r32^2 + r33^2))
        # tan(yaw)  = r21/r11
        # np.arctan2(y, x)
        
        roll = np.arctan2(T06[2,1],T06[2,2])
        pitch= np.arctan2(-T06[2,0],np.sqrt(T06[2,1]*T06[2,1] + T06[2,2]*T06[2,2]))
        yaw  = np.arctan2(T06[1,0],T06[0,0])

        x_current = np.array([[T06[0, 3]],
                              [T06[1, 3]],
                              [T06[2, 3]],
                                   [roll],
                                  [pitch],
                                    [yaw]])

        return x_current

    def jacobian(self,theta):

        A1 = self.dh_transformation(theta[0,0],self.alpha[0,0],self.d[0,0],self.a[0,0])
        A2 = self.dh_transformation(theta[1,0],self.alpha[1,0],self.d[1,0],self.a[1,0])
        A3 = self.dh_transformation(theta[2,0],self.alpha[2,0],self.d[2,0],self.a[2,0])
        A4 = self.dh_transformation(theta[3,0],self.alpha[3,0],self.d[3,0],self.a[3,0])
        A5 = self.dh_transformation(theta[4,0],self.alpha[4,0],self.d[4,0],self.a[4,0])
        A6 = self.dh_transformation(theta[5,0],self.alpha[5,0],self.d[5,0],self.a[5,0])

        T01 = A1
        T02 = A1 @ A2
        T03 = A1 @ A2 @ A3
        T04 = A1 @ A2 @ A3 @ A4
        T05 = A1 @ A2 @ A3 @ A4 @ A5
        T06 = A1 @ A2 @ A3 @ A4 @ A5 @ A6

        Z0 = np.array([[0],
                       [0],
                       [1]])

        Z1 = np.array([[T01[0,2]],
                       [T01[1,2]],
                       [T01[2,2]]])

        Z2 = np.array([[T02[0,2]],
                       [T02[1,2]],
                       [T02[2,2]]])

        Z3 = np.array([[T03[0,2]],
                       [T03[1,2]],
                       [T03[2,2]]])

        Z4 = np.array([[T04[0,2]],
                       [T04[1,2]],
                       [T04[2,2]]])

        Z5 = np.array([[T05[0,2]],
                       [T05[1,2]],
                       [T05[2,2]]])

        O0 = np.array([[0],
                       [0],
                       [0]])

        O1 = np.array([[T01[3,0]],
                       [T01[3,1]],
                       [T01[3,2]]])

        O2 = np.array([[T02[3,0]],
                       [T02[3,1]],
                       [T02[3,2]]])

        O3 = np.array([[T03[3,0]],
                       [T03[3,1]],
                       [T03[3,2]]])

        O4 = np.array([[T04[3,0]],
                       [T04[3,1]],
                       [T04[3,2]]])

        O5 = np.array([[T05[3,0]],
                       [T05[3,1]],
                       [T05[3,2]]])

        O6 = np.array([[T06[3,0]],
                       [T06[3,1]],
                       [T06[3,2]]])

        Jv1 = np.transpose(np.cross(np.transpose(Z0),np.transpose(O6-O0))) 
        Jv2 = np.transpose(np.cross(np.transpose(Z1),np.transpose(O6-O1))) 
        Jv3 = np.transpose(np.cross(np.transpose(Z2),np.transpose(O6-O2))) 
        Jv4 = np.transpose(np.cross(np.transpose(Z3),np.transpose(O6-O3))) 
        Jv5 = np.transpose(np.cross(np.transpose(Z4),np.transpose(O6-O4))) 
        Jv6 = np.transpose(np.cross(np.transpose(Z5),np.transpose(O6-O5)))

        Jw1 = Z0
        Jw2 = Z1
        Jw3 = Z2
        Jw4 = Z3
        Jw5 = Z4
        Jw6 = Z5

        J1 = np.append(Jv1,Jw1,axis=0)
        J2 = np.append(Jv2,Jw2,axis=0)
        J3 = np.append(Jv3,Jw3,axis=0)
        J4 = np.append(Jv4,Jw4,axis=0)
        J5 = np.append(Jv5,Jw5,axis=0)
        J6 = np.append(Jv6,Jw6,axis=0)

        J = np.append(np.append(np.append(np.append(np.append(J1,J2,axis=1),J3,axis=1),J4,axis=1),J5,axis=1),J6,axis=1)

        return J

    def jacobian_analytical(self,theta,roll,pitch,yaw):

        B = np.array([[np.cos(yaw)*np.sin(pitch), -np.sin(yaw), 0],
                      [np.sin(yaw)*np.sin(pitch),  np.cos(yaw), 0],
                      [            np.cos(pitch),            0, 1]])

        Binv = np.linalg.inv(B)

        Ja_mul = np.array([[1, 0, 0,         0,         0,         0],
                           [0, 1, 0,         0,         0,         0],
                           [0, 0, 1,         0,         0,         0],
                           [0, 0, 0, Binv[0,0], Binv[0,1], Binv[0,2]],
                           [0, 0, 0, Binv[1,0], Binv[1,1], Binv[1,2]],
                           [0, 0, 0, Binv[2,0], Binv[2,1], Binv[2,2]]])

        Ja = Ja_mul @ self.jacobian(theta)

        return Ja