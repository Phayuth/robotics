import numpy as np
import matplotlib.pyplot as plt

class planar_rr:
    def __init__(self):
        self.alpha1 = 0
        self.alpha2 = 0
        self.d1 = 0
        self.d2 = 0
        self.a1 = 1
        self.a2 = 1

    def dh_transformation(theta,alpha,d,a):
        R = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                      [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                      [            0,                np.sin(alpha),                np.cos(alpha),               d],
                      [            0,                            0,                            0,               1]])
        return R

    def forward_kinematic(self, theta):
        theta1 = theta[0,0]
        theta2 = theta[1,0]

        x = self.a1*np.cos(theta1) + self.a2*np.cos(theta1+theta2)
        y = self.a1*np.sin(theta1) + self.a2*np.sin(theta1+theta2)

        x = np.array([[x],[y]])

        return x

    def jacobian(self, theta):
        theta1 = theta[0,0]
        theta2 = theta[1,0]

        J = np.array([[-self.a1*np.sin(theta1)-self.a2*np.sin(theta1+theta2) , -self.a2*np.sin(theta1+theta2)],
                       [self.a1*np.cos(theta1)+self.a2*np.cos(theta1+theta2) ,  self.a2*np.cos(theta1+theta2)]])
        return J

    def forward_kinematic_dh(self,theta):
        theta1 = theta[0,0]
        theta2 = theta[1,0]

        A1 = self.dh_transformation(theta1,self.alpha1,self.d1,self.a1)
        A2 = self.dh_transformation(theta2,self.alpha2,self.d2,self.a2)

        T = A1 @ A2
        p2 = np.array([[0],[0],[0],[1]]) # the fourth element MUST be equal to 1
        p0 = T @ p2

        return p0

    def jacobian_dh(self,theta):
        theta1 = theta[0,0]
        theta2 = theta[1,0]

        O0 = np.array([[0],
                       [0],
                       [0]])

        O1 = np.array([[self.a1*np.cos(theta1)],
                       [self.a1*np.sin(theta1)],
                       [      0              ]])

        O2 = np.array([[self.a1*np.cos(theta1) + self.a2*np.cos(theta1+theta2)],
                       [self.a1*np.sin(theta1) + self.a2*np.sin(theta1+theta2)],
                       [                       0                             ]])

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

        J1 = np.append(Jv1,Jw1,axis=0) # if not use axis = the result is 1x6, axis=0 the result is 6x1, axis=1 the result is 3x2
        J2 = np.append(Jv2,Jw2,axis=0)

        J = np.append(J1,J2,axis=1) # we want J = [J1 , J2] side by side => use axis = 1

        return J
    
    def plot_arm(self, theta, *args, **kwargs):
        theta1 = theta[0,0]
        theta2 = theta[1,0]

        shoulder = np.array([0, 0])
        elbow = shoulder + np.array([self.a1 * np.cos(theta1), self.a1 * np.sin(theta1)])
        wrist = elbow + np.array([self.a2 * np.cos(theta1 + theta2), self.a2 * np.sin(theta1 + theta2)])

        plt.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]], 'k-')
        plt.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], 'k-')

        plt.plot(shoulder[0], shoulder[1], 'ro')
        plt.plot(elbow[0], elbow[1], 'ro')
        plt.plot(wrist[0], wrist[1], 'ro')
        
        
        title = kwargs.get('title', None)
        plt.annotate("X pos = "+str(wrist[0]), xy=(0, 1.8+2), xycoords="data",va="center", ha="center")
        plt.annotate("Y pos = "+str(wrist[1]), xy=(0, 1.5+2), xycoords="data",va="center", ha="center")
        
        circle1 = plt.Circle((0, 0), self.a1+self.a2,alpha=0.5, edgecolor='none')
        plt.gca().add_patch(circle1)
        plt.title(title)
        
        plt.xlim(-3, 3)
        plt.ylim(-2, 4)

        plt.show()