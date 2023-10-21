import numpy as np


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


# UR5 copsim
# thetaInit = np.array([0, 0, 0, 0, 0, 0]).reshape(6, 1)
# thetaGoal = np.array([1.2, 0, -0.2, 0, -1.2, -0.2]).reshape(6, 1)
# thetaApp = np.array([-1.57, -2.3, -1.4, 0.1, 1.57, 0]).reshape(6, 1)

# sim
simThetaInit = np.array([np.deg2rad(-27.68), np.deg2rad(2.95), np.deg2rad(-26.58), np.deg2rad(-15.28), np.deg2rad(87.43), np.deg2rad(0.0)]).reshape(6, 1)
simThetaApp = np.array([np.deg2rad(-27.68), np.deg2rad(2.95), np.deg2rad(-26.58), np.deg2rad(-15.28), np.deg2rad(87.43), np.deg2rad(0.0)]).reshape(6, 1)
simThetaGoal = np.array([np.deg2rad(-27.68), np.deg2rad(2.95), np.deg2rad(-26.58), np.deg2rad(-15.28), np.deg2rad(87.43), np.deg2rad(0.0)]).reshape(6, 1)

# Init
thetaInit = np.array([np.deg2rad(-27.68), np.deg2rad(2.95), np.deg2rad(-26.58), np.deg2rad(-15.28), np.deg2rad(87.43), np.deg2rad(0.0)]).reshape(6, 1)

# Successful Auxilary Pose and Goal Pose
# Cand 1
thetaGoal1 = np.array([np.deg2rad(124.02), np.deg2rad(-70.58), np.deg2rad(-91.21), np.deg2rad(-50.84), np.deg2rad(105.67), np.deg2rad(-0.02)]).reshape(6, 1)
thetaApp1 = np.array([np.deg2rad(124.02), np.deg2rad(-56.47), np.deg2rad(-91.21), np.deg2rad(-59.58), np.deg2rad(103.13), np.deg2rad(-0.08)]).reshape(6, 1)

# cand 2
thetaGoal2 = np.array([np.deg2rad(125.95), np.deg2rad(-63.79), np.deg2rad(-112.1), np.deg2rad(201.81), np.deg2rad(-99.11), np.deg2rad(0.16)]).reshape(6, 1)
thetaApp2 = np.array([np.deg2rad(125.90), np.deg2rad(-54.71), np.deg2rad(-111.6), np.deg2rad(181.35), np.deg2rad(-98.61), np.deg2rad(0.14)]).reshape(6, 1)

# From ROS
# Cand 1
qCurrent = np.array([-0.3513484733244089, -0.8841841545935455, -1.7460461905029725, -0.3149857471639031, 0.8279524666993007, -0.023037786917641353]).reshape(6, 1)
qGoal = np.array([5.229049405606208, 4.005793026442909, -0.12450149024625597, -0.18886354720065374, 2.103094506316899, -5.047631697794035]).reshape(6, 1)
qAux = np.array([5.119112501615053, 4.348336564349895, -0.8201569789671345, 0.20305320620760767, -4.08750436527597, 1.306417971784997]).reshape(6, 1)

# New
newThetaInit = np.array([np.deg2rad(-0.39), np.deg2rad(-5.96), np.deg2rad(-3.43), np.deg2rad(6.08), np.deg2rad(1.68), np.deg2rad(-3.43)]).reshape(6, 1)
newThetaApp = np.array([np.deg2rad(-107.49), np.deg2rad(-64.44), np.deg2rad(-118.96), np.deg2rad(10.21), np.deg2rad(92.49), np.deg2rad(-3.45)]).reshape(6, 1)
newThetaGoal = np.array([np.deg2rad(-107.48), np.deg2rad(-74.36), np.deg2rad(-111.06), np.deg2rad(12.07), np.deg2rad(95.00), np.deg2rad(-3.47)]).reshape(6, 1)

# 3dof only
new3dofThetaInit = np.array([np.deg2rad(-0.39), np.deg2rad(-5.96), np.deg2rad(-3.43)]).reshape(3, 1)
new3dofThetaApp = np.array([np.deg2rad(-107.49), np.deg2rad(-64.44), np.deg2rad(-118.96)]).reshape(3, 1)
new3dofThetaGoal = np.array([np.deg2rad(-107.48), np.deg2rad(-74.36), np.deg2rad(-111.06)]).reshape(3, 1)

if __name__ == "__main__":
    wrapped_angle = wrap_to_pi(qGoal)
    print(wrapped_angle)