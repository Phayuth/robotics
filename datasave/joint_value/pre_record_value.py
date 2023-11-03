import numpy as np


class SinglePose:

    class Pose1: # UR5 copsim
        thetaInit = np.array([0, 0, 0, 0, 0, 0]).reshape(6, 1)
        thetaApp = np.array([-1.57, -2.3, -1.4, 0.1, 1.57, 0]).reshape(6, 1)
        thetaGoal = np.array([1.2, 0, -0.2, 0, -1.2, -0.2]).reshape(6, 1)

    class Pose2:
        thetaInit = np.array([np.deg2rad(-27.68), np.deg2rad(2.95), np.deg2rad(-26.58), np.deg2rad(-15.28), np.deg2rad(87.43), np.deg2rad(0.0)]).reshape(6, 1)
        thetaApp = np.array([np.deg2rad(-27.68), np.deg2rad(2.95), np.deg2rad(-26.58), np.deg2rad(-15.28), np.deg2rad(87.43), np.deg2rad(0.0)]).reshape(6, 1)
        thetaGoal = np.array([np.deg2rad(-27.68), np.deg2rad(2.95), np.deg2rad(-26.58), np.deg2rad(-15.28), np.deg2rad(87.43), np.deg2rad(0.0)]).reshape(6, 1)

    class Pose3:
        thetaInit = np.array([np.deg2rad(-27.68), np.deg2rad(2.95), np.deg2rad(-26.58), np.deg2rad(-15.28), np.deg2rad(87.43), np.deg2rad(0.0)]).reshape(6, 1)
        thetaApp = np.array([np.deg2rad(124.02), np.deg2rad(-56.47), np.deg2rad(-91.21), np.deg2rad(-59.58), np.deg2rad(103.13), np.deg2rad(-0.08)]).reshape(6, 1)
        thetaGoal = np.array([np.deg2rad(124.02), np.deg2rad(-70.58), np.deg2rad(-91.21), np.deg2rad(-50.84), np.deg2rad(105.67), np.deg2rad(-0.02)]).reshape(6, 1)

    class Pose4:
        thetaInit = np.array([np.deg2rad(-27.68), np.deg2rad(2.95), np.deg2rad(-26.58), np.deg2rad(-15.28), np.deg2rad(87.43), np.deg2rad(0.0)]).reshape(6, 1)
        thetaApp = np.array([np.deg2rad(125.90), np.deg2rad(-54.71), np.deg2rad(-111.6), np.deg2rad(181.35), np.deg2rad(-98.61), np.deg2rad(0.14)]).reshape(6, 1)
        thetaGoal = np.array([np.deg2rad(125.95), np.deg2rad(-63.79), np.deg2rad(-112.1), np.deg2rad(201.81), np.deg2rad(-99.11), np.deg2rad(0.16)]).reshape(6, 1)

    class Pose5:
        thetaInit = np.array([-0.3513484733244089, -0.8841841545935455, -1.7460461905029725, -0.3149857471639031, 0.8279524666993007, -0.023037786917641353]).reshape(6, 1)
        thetaApp = np.array([5.119112501615053, 4.348336564349895, -0.8201569789671345, 0.20305320620760767, -4.08750436527597, 1.306417971784997]).reshape(6, 1)
        thetaGoal = np.array([5.229049405606208, 4.005793026442909, -0.12450149024625597, -0.18886354720065374, 2.103094506316899, -5.047631697794035]).reshape(6, 1)

    class Pose6: # new used 
        thetaInit = np.array([np.deg2rad(-0.39), np.deg2rad(-5.96), np.deg2rad(-3.43), np.deg2rad(6.08), np.deg2rad(1.68), np.deg2rad(-3.43)]).reshape(6, 1)
        thetaApp = np.array([np.deg2rad(-107.49), np.deg2rad(-64.44), np.deg2rad(-118.96), np.deg2rad(10.21), np.deg2rad(92.49), np.deg2rad(-3.45)]).reshape(6, 1)
        thetaGoal = np.array([np.deg2rad(-107.48), np.deg2rad(-74.36), np.deg2rad(-111.06), np.deg2rad(12.07), np.deg2rad(95.00), np.deg2rad(-3.47)]).reshape(6, 1)

    class Pose7:
        thetaInit3dof = np.array([np.deg2rad(-0.39), np.deg2rad(-5.96), np.deg2rad(-3.43)]).reshape(3, 1)
        thetaApp3dof = np.array([np.deg2rad(-107.49), np.deg2rad(-64.44), np.deg2rad(-118.96)]).reshape(3, 1)
        thetaGoal3dof = np.array([np.deg2rad(-107.48), np.deg2rad(-74.36), np.deg2rad(-111.06)]).reshape(3, 1)


class MultiplePoses:

    class Pose1:
        pass


class PreRecordedPath:

    path = np.array([[-0.00680678, -0.10402162, -0.05986479,  0.10611602,  0.02932153, -0.05986479],
                     [-0.17498055, -0.19584984, -0.24127577,  0.11260115,  0.17191593, -0.0598962 ],
                     [-0.33829331, -0.43804178, -0.56917319, -0.01237059,  0.27132088, -0.16652974],
                     [-0.33199506, -0.59094074, -0.71948546, -0.14552028,  0.37320614, -0.2925417 ],
                     [-0.68927399, -0.69681341, -0.81629521, -0.09467394,  0.38986462, -0.45615326],
                     [-0.86116716, -0.64265464, -0.95609087,  0.05480456,  0.48548210, -0.53671516],
                     [-1.01276666, -0.71465904, -1.12341449,  0.07323656,  0.65409336, -0.46553744],
                     [-1.16436617, -0.78666344, -1.29073811,  0.09166856,  0.82270461, -0.39435972],
                     [-1.31596567, -0.85866784, -1.45806173,  0.11010056,  0.99131587, -0.323182  ],
                     [-1.46756518, -0.93067224, -1.62538535,  0.12853256,  1.15992713, -0.25200428],
                     [-1.87605441, -1.12469017, -2.07624368,  0.17819812,  1.61425503, -0.06021386],
                     [-1.87587988, -1.29782683, -1.93836267,  0.21066124,  1.65806279, -0.06056293]])
    

if __name__ == "__main__":
    q = SinglePose.Pose1()
    qS = q.thetaInit
    qA = q.thetaApp
    qG = q.thetaGoal