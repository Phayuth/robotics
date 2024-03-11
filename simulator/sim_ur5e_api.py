import sys
import os

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))
sys.path.append("/home/yuth/coppelia/programming/zmqRemoteApi/clients/python")

import numpy as np
from zmqRemoteApi import RemoteAPIClient


class UR5eArmCoppeliaSimAPI:

    def __init__(self, isVisual=False):
        # physical properties, limit joint, dof
        self.configLimit = [[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]]
        self.configDoF = len(self.configLimit)

        # joint max
        self.jmVel = np.deg2rad(180)  # 180 deg per sec
        self.jmAccel = np.deg2rad(150)
        self.jmJerk = np.deg2rad(100)
        self.jmaxVel = [self.jmVel] * self.configDoF
        self.jmaxAccel = [self.jmAccel] * self.configDoF
        self.jmaxJerk = [self.jmJerk] * self.configDoF

        # tip max
        self.tmaxVel = [0.1]
        self.tmaxAccel = [0.01]
        self.tmaxJerk = [80]

        # coppeliasim
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject("sim")
        # self.client.setStepping(True)

        # model
        self.isVisual = isVisual
        self.robotHeadNames = ["/UR5e_dynamic", "/UR5e_virtual", "/UR5e_goal", "/UR5e_aux"]
        # self.modelBase = [self.sim.getObject(robotModel) for robotModel in self.robotHeadNames]

        # arm joint handle
        self.jointNames = ["/shoulder_pan_joint",
                           "/shoulder_link_resp/shoulder_lift_joint",
                           "/upper_arm_link_resp/elbow_joint",
                           "/forearm_link_resp/wrist_1_joint",
                           "/wrist_1_link_resp/wrist_2_joint",
                           "/wrist_2_link_resp/wrist_3_joint"]

        self.jointDynamicHandles = [self.sim.getObject(self.robotHeadNames[0] + "".join(self.jointNames[0 : i + 1])) for i in range(len(self.jointNames))]
        self.jointVirtualHandles = [self.sim.getObject(self.robotHeadNames[1] + "".join(self.jointNames[0 : i + 1])) for i in range(len(self.jointNames))]

        if self.isVisual:
            self.jointGoalHandles = [self.sim.getObject(self.robotHeadNames[2] + "".join(self.jointNames[0 : i + 1])) for i in range(len(self.jointNames))]
            self.jointAuxHandles = [self.sim.getObject(self.robotHeadNames[3] + "".join(self.jointNames[0 : i + 1])) for i in range(len(self.jointNames))]

        # arm collision handle
        self.robotBase = self.sim.getObject("/UR5e_virtual")
        self.collection = self.sim.createCollection(0)
        self.sim.addItemToCollection(self.collection, self.sim.handle_tree, self.robotBase, 0)

        # arm IK
        self.simIK = self.client.getObject("simIK")
        self.simBase = self.sim.getObject("/UR5e_dynamic")
        self.simWorldFrame = -1
        self.simTip = self.sim.getObject(self.robotHeadNames[0] + "".join(self.jointNames) + "/wrist_3_link_resp/tip")
        self.simTarget = self.sim.getObject("/UR5e_dynamic/target")

        self.ikEnv = self.simIK.createEnvironment()
        self.ikPrecision = [0.00005, np.deg2rad(0.1)]
        self.fcThresholdDist = 0.5  # 0.65
        self.fcMaxTime = 0.01
        self.fcMetric = [1, 1, 1, 0.1]

        # arm IK Group 1
        self.ikGroup = self.simIK.createIkGroup(self.ikEnv)
        self.ikElement, simToIkObjectMapping = self.simIK.addIkElementFromScene(self.ikEnv, self.ikGroup, self.simBase, self.simTip, self.simTarget, self.simIK.constraint_pose)
        self.simIK.setIkElementPrecision(self.ikEnv, self.ikGroup, self.ikElement, self.ikPrecision)

        # arm IK Group 2 (PseudoInv)
        self.ikGroupUndamped = self.simIK.createIkGroup(self.ikEnv)
        self.ikElementUndamped, simToIkObjectMappingUndamped = self.simIK.addIkElementFromScene(self.ikEnv, self.ikGroupUndamped, self.simBase, self.simTip, self.simTarget, self.simIK.constraint_pose)
        self.simIK.setIkGroupCalculation(self.ikEnv, self.ikGroupUndamped, self.simIK.method_pseudo_inverse, 0, 99)  # dampcte, maxIteration
        self.simIK.setIkElementPrecision(self.ikEnv, self.ikGroupUndamped, self.ikElementUndamped, self.ikPrecision)

        # arm IK Group 3 (DLSQ)
        self.ikGroupDamped = self.simIK.createIkGroup(self.ikEnv)
        self.ikElementdamped, simToIkObjectMappingdamped = self.simIK.addIkElementFromScene(self.ikEnv, self.ikGroupDamped, self.simBase, self.simTip, self.simTarget, self.simIK.constraint_pose)
        self.simIK.setIkGroupCalculation(self.ikEnv, self.ikGroupDamped, self.simIK.method_damped_least_squares, 0.5, 99)  # dampcte, maxIteration
        self.simIK.setIkElementPrecision(self.ikEnv, self.ikGroupDamped, self.ikElementdamped, self.ikPrecision)

        self.ikJointHandles = [simToIkObjectMapping[id] for id in self.jointDynamicHandles]
        self.ikTarget = simToIkObjectMapping[self.simTarget]
        self.ikBase = simToIkObjectMapping[self.simBase]

    def start_sim(self):
        self.sim.startSimulation()

    def stop_sim(self):
        self.sim.stopSimulation()

    def step_sim(self):
        self.client.step()

    def collision_check(self, jointValue):
        self.set_joint_position(self.jointVirtualHandles, jointValue)
        result, _ = self.sim.checkCollision(self.collection, self.sim.handle_all)
        if result == 1:
            return True
        else:
            return False

    def get_joint_position(self, jointHandle):
        return [self.sim.getJointPosition(jH) for jH in jointHandle]

    def set_joint_position(self, jointHandle, jointValue):
        for i in range(self.configDoF):
            self.sim.setJointPosition(jointHandle[i], jointValue[i, 0])

    def set_joint_target_position(self, jointHandle, jointValue):
        for i in range(self.configDoF):
            self.sim.setJointTargetPosition(jointHandle[i], jointValue[i, 0])

    def set_pose_aux_goal(self, auxValue=None, goalValue=None):
        if auxValue is not None:
            self.set_joint_position(self.jointAuxHandles, auxValue)
        if goalValue is not None:
            self.set_joint_position(self.jointGoalHandles, goalValue)

    def update_joint_position(self, qReal=None):
        """
        Set real joint position to simulate robot along with IK joint.
        Call update before any IK operation is performed.

        Parameters
        ----------
        qReal : List
            Real joint postion from sensor reading.
        """
        if qReal is not None:
            self.set_joint_position(self.jointDynamicHandles, np.array(qReal).reshape(6, 1))

        qg = self.get_joint_position(self.jointDynamicHandles)
        for i in range(6):
            self.simIK.setJointPosition(self.ikEnv, self.ikJointHandles[i], qg[i])

    def ik_find_config(self, targetPose):
        """
        Find joint position given `targetPose` at random. It is used when motion is far from current pose.

        Parameters
        ----------
        targetPose : List
            End-effector pose.
            Ex : [x,y,z,qx,qy,qz,qw]

        Returns
        -------
        List
            Joint position from IK.
        """
        self.simIK.setObjectPose(self.ikEnv, self.ikTarget, self.ikBase, targetPose)
        state = self.simIK.findConfig(self.ikEnv, self.ikGroup, self.ikJointHandles, self.fcThresholdDist, self.fcMaxTime, self.fcMetric)
        if state is not None:
            return state

    def ik_solve(self, targetPose):
        """
        Solve IK with Jacobian Based Solver. It is used when motion is near to current pose.

        Parameters
        ----------
        targetPose : List
            End-effector pose.
            Ex : [x,y,z,qx,qy,qz,qw]

        Returns
        -------
        List
            Joint position from IK.
        """
        self.simIK.setObjectPose(self.ikEnv, self.ikTarget, self.ikBase, targetPose)

        resultUndamped = self.simIK.handleIkGroup(self.ikEnv, self.ikGroupUndamped)
        if resultUndamped != self.simIK.result_success:
            resultDamped = self.simIK.handleIkGroup(self.ikEnv, self.ikGroupDamped)
            # if resultDamped != self.simIK.result_success:
            #     return None
            # else:
            #     state = [self.simIK.getJointPosition(self.ikEnv, self.ikJointHandles[i]) for i in range(self.configDoF)]
            #     return state
        state = [self.simIK.getJointPosition(self.ikEnv, self.ikJointHandles[i]) for i in range(self.configDoF)]
        return state

    def move_to_config_sequences(self, movement, movementVelo):
        """
        Move robot in a sequence of configuration space. It is used for long sequence of motion.
        Stop at each knot if `movementVelo` is not given otherwise move smoothly.
        Blocking Call function.

        Parameters
        ----------
        movement : List[List]
            Sequence of joint position to be moved.
        movementVelo : List[List]
            Sequence of joint velocity at knot point.

        Returns
        -------
        List[List]
            Sequence of joint position history for every time interval.
        """
        qSequenceHistory = []

        def movCallback(config, vel, accel, handles):
            qSequenceHistory.append(config)
            self.set_joint_position(handles, np.array(config).reshape(6, 1))

        currentPos = self.get_joint_position(self.jointDynamicHandles)
        currentVel = None
        currentAccel = None
        for mi, m in enumerate(movement):
            currentPos, currentVel, currentAccel, timeLeft = self.sim.moveToConfig(-1, currentPos, currentVel, currentAccel, self.jmaxVel, self.jmaxAccel, self.jmaxJerk, m, movementVelo[mi], movCallback, self.jointDynamicHandles)

        return qSequenceHistory

    def move_to_pose_sequences(self, movement):
        """
        Move robot in a sequence of end-effector task space. It is used for long sequence of motion.
        Stop at each knot.
        Blocking call function.

        Parameters
        ----------
        movement : List[List]
            Sequence of end-effector pose to be moved.
            Ex : [x,y,z,qx,qy,qz,qw]

        Returns
        -------
        Tuple[List, List]
            Sequence of joint position history for every time interval.
            Sequence of joint position at Knot point.
        """
        qSequenceHistory = []
        qKnotHistory = []

        def callback(pose, vel, accel, handle):
            self.sim.setObjectPose(handle, self.simBase, pose)
            q = self.ik_solve(pose)
            if q is not None:
                qSequenceHistory.append(q)
                self.set_joint_position(self.jointDynamicHandles, np.array(q).reshape(6, 1))

        currentPose = self.sim.getObjectPose(self.simTip, self.simBase)
        for mi, m in enumerate(movement):
            currentPose, timeLeft = self.sim.moveToPose(-1, currentPose, self.jmaxVel, self.jmaxAccel, self.jmaxJerk, m, callback, self.simTarget, [1, 1, 1, 0.1])
            qKnot = self.get_joint_position(self.jointDynamicHandles)
            qKnotHistory.append(qKnot)

        return qSequenceHistory, qKnotHistory

    def play_back_path(self, pathArray):
        """
        Moved robot in joint space. It is used only for visualization purpose.

        Parameters
        ----------
        pathArray : np.array
            Joint position to be moved. In shape of (numDoF, numSeq).
        """
        if self.isVisual:
            self.set_pose_aux_goal(auxValue=pathArray[:, -2, np.newaxis], goalValue=pathArray[:, -1, np.newaxis])

        path = pathArray.T.flatten().tolist()
        tfFromStart = 5.0
        times = np.linspace(0, tfFromStart, pathArray.shape[1]).tolist()
        startTime = self.sim.getSimulationTime()
        t = 0.0
        while t < times[-1]:
            conf = self.sim.getPathInterpolatedConfig(path, times, t)
            self.set_joint_position(self.jointDynamicHandles, np.array(conf).reshape(6, 1))
            t = self.sim.getSimulationTime() - startTime


if __name__ == "__main__":
    from datasave.joint_value.pre_record_value import PreRecordedPath
    from datasave.joint_value.experiment_paper import URHarvesting

    ur = UR5eArmCoppeliaSimAPI()

    ur.start_sim()
    try:
        while True:
            # set joint visualize
            q = URHarvesting.PoseSingle1()
            qS = q.xStart
            qA = q.xApp
            qG = q.xGoal
            ur.set_pose_aux_goal(qA, qG)

            # play back
            path = PreRecordedPath.path.T  # shape(6, 12)
            ur.play_back_path(path)

    except KeyboardInterrupt:
        pass

    finally:
        ur.stop_sim()