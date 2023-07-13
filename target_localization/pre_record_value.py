import numpy as np

# UR5 copsim
# thetaInit = np.array([0, 0, 0, 0, 0, 0]).reshape(6, 1)
# thetaGoal = np.array([1.2, 0, -0.2, 0, -1.2, -0.2]).reshape(6, 1)
# thetaApp = np.array([-1.57, -2.3, -1.4, 0.1, 1.57, 0]).reshape(6, 1)

# Init
thetaInit = np.array([np.deg2rad(-27.68), np.deg2rad(2.95), np.deg2rad(-26.58), np.deg2rad(-15.28), np.deg2rad(87.43), np.deg2rad(0.0)]).reshape(6, 1)

# Successful Auxilary Pose and Goal Pose
# Cand 1
thetaGoal1 = np.array([np.deg2rad(124.02), np.deg2rad(-70.58), np.deg2rad(-91.21), np.deg2rad(-50.84), np.deg2rad(105.67), np.deg2rad(-0.02)]).reshape(6, 1)
thetaApp1 = np.array([np.deg2rad(124.02), np.deg2rad(-56.47), np.deg2rad(-91.21), np.deg2rad(-59.58), np.deg2rad(103.13), np.deg2rad(-0.08)]).reshape(6, 1)

# cand 2
thetaGoal2 = np.array([np.deg2rad(125.95), np.deg2rad(-63.79), np.deg2rad(-112.1), np.deg2rad(201.81), np.deg2rad(-99.11), np.deg2rad(0.16)]).reshape(6, 1)
thetaApp2 = np.array([np.deg2rad(125.90), np.deg2rad(-54.71), np.deg2rad(-111.6), np.deg2rad(181.35), np.deg2rad(-98.61), np.deg2rad(0.14)]).reshape(6, 1)

# cand 3
thetaGoal3 = np.array([np.deg2rad(109.16), np.deg2rad(-72.75), np.deg2rad(-100.1), np.deg2rad(195.48), np.deg2rad(-130.7), np.deg2rad(0.16)]).reshape(6, 1)
thetaApp3 = np.array([np.deg2rad(102.59), np.deg2rad(-67.46), np.deg2rad(-105.1), np.deg2rad(192.53), np.deg2rad(-133.9), np.deg2rad(0.16)]).reshape(6, 1)

# cand 4
thetaGoal4 = np.array([np.deg2rad(172.60), np.deg2rad(-75.28), np.deg2rad(-83.96), np.deg2rad(159.51), np.deg2rad(-6.16), np.deg2rad(-141.6)]).reshape(6, 1)
thetaApp4 = np.array([np.deg2rad(186.18), np.deg2rad(-74.60), np.deg2rad(-83.96), np.deg2rad(153.77), np.deg2rad(12.48), np.deg2rad(-141.7)]).reshape(6, 1)

# cand 5
thetaGoal5 = np.array([np.deg2rad(147.29), np.deg2rad(-69.53), np.deg2rad(-128.8), np.deg2rad(251.83), np.deg2rad(-75.48), np.deg2rad(-183.9)]).reshape(6, 1)
thetaApp5 = np.array([np.deg2rad(139.86), np.deg2rad(-60.75), np.deg2rad(-137.7), np.deg2rad(245.12), np.deg2rad(-77.17), np.deg2rad(-193.3)]).reshape(6, 1)

# cand 6
thetaGoal6 = np.array([np.deg2rad(183.38), np.deg2rad(-90.38), np.deg2rad(-131.35), np.deg2rad(-39.44), np.deg2rad(26.22), np.deg2rad(37.62)]).reshape(6, 1)
thetaApp6 = np.array([np.deg2rad(193.48), np.deg2rad(-81.39), np.deg2rad(-131.70), np.deg2rad(-69.87), np.deg2rad(20.59), np.deg2rad(44.59)]).reshape(6, 1)

# Fail Auxilary Pose and Goal Pose
