import sys

sys.path.append("/home/yuth/coppelia/programming/zmqRemoteApi/clients/python")
import numpy as np
from zmqRemoteApi import RemoteAPIClient

print('Program started')

# create client and sim object
client = RemoteAPIClient()
sim = client.getObject('sim')
simIK = client.getObject('simIK')


def solving():
    jointNames = ['/shoulder_pan_joint',
                '/shoulder_link_resp/shoulder_lift_joint',
                '/upper_arm_link_resp/elbow_joint',
                '/forearm_link_resp/wrist_1_joint',
                '/wrist_1_link_resp/wrist_2_joint',
                '/wrist_2_link_resp/wrist_3_joint']

    # arm joint handle
    jointHandleList = [sim.getObject('/' + 'UR5e_dynamic' + ''.join(jointNames[0:i+1])) for i in range(len(jointNames))]

    simBase=sim.getObject('/UR5e_dynamic')
    simTip=sim.getObject('/UR5e_dynamic/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint/wrist_1_link_resp/wrist_2_joint/wrist_2_link_resp/wrist_3_joint/wrist_3_link_resp/tip')
    simTarget=sim.getObject('/UR5e_dynamic/target')

    # ik env
    ikEnv=simIK.createEnvironment()

    # solve for nearest inverse configuration
    ikGroupUndamped=simIK.createIkGroup(ikEnv)
    ikElementUndamped, simToIkObjectMappingUndamped = simIK.addIkElementFromScene(ikEnv,ikGroupUndamped,simBase,simTip,simTarget,simIK.constraint_pose)
    simIK.setIkGroupCalculation(ikEnv,ikGroupUndamped,simIK.method_pseudo_inverse,0,10)
    simIK.setIkElementPrecision(ikEnv,ikGroupUndamped,ikElementUndamped,[0.00005,0.1*np.pi/180])

    ikGroupDamped=simIK.createIkGroup(ikEnv)
    ikElementdamped, simToIkObjectMappingdamped = simIK.addIkElementFromScene(ikEnv,ikGroupDamped,simBase,simTip,simTarget,simIK.constraint_pose)
    simIK.setIkGroupCalculation(ikEnv,ikGroupDamped,simIK.method_damped_least_squares,0.3,99)
    simIK.setIkElementPrecision(ikEnv,ikGroupDamped,ikElementdamped,[0.00005,0.1*np.pi/180])

    #
    ikJointHandles = [simToIkObjectMappingdamped[id] for id in jointHandleList] # undamped or damped ik joint is the same. pick whatever
    ikTarget=simToIkObjectMappingdamped[simTarget]
    ikBase=simToIkObjectMappingdamped[simBase]

    def setjoint_value(state):
        for i in range(len(state)):
            sim.setJointPosition(jointHandleList[i], state[i])

    sim.startSimulation()

    try:
        while True:
            targets=sim.getObject('/testTarget5')
            targetTranformationMatrix = sim.getObjectMatrix(targets, simBase)
            simIK.setObjectMatrix(ikEnv,ikTarget,ikBase, targetTranformationMatrix)

            resultUndamped=simIK.handleIkGroup(ikEnv, ikGroupUndamped)
            print(f"> resultUndamped: {resultUndamped}")
            if resultUndamped != simIK.result_success:
                resultDamped=simIK.handleIkGroup(ikEnv, ikGroupDamped)
                print(f"> resultDamped: {resultDamped}")

            state = [simIK.getJointPosition(ikEnv, ikJointHandles[i]) for i in range(6)]
            setjoint_value(state)

    finally:
        sim.stopSimulation()

# solving()

# =====================================================================

def finding_config():
    jointNames = ['/shoulder_pan_joint',
                '/shoulder_link_resp/shoulder_lift_joint',
                '/upper_arm_link_resp/elbow_joint',
                '/forearm_link_resp/wrist_1_joint',
                '/wrist_1_link_resp/wrist_2_joint',
                '/wrist_2_link_resp/wrist_3_joint']

    # arm joint handle
    jointHandleList = [sim.getObject('/' + 'UR5e_dynamic' + ''.join(jointNames[0:i+1])) for i in range(len(jointNames))]

    simBase=sim.getObject('/UR5e_dynamic')
    simTip=sim.getObject('/UR5e_dynamic/shoulder_pan_joint/shoulder_link_resp/shoulder_lift_joint/upper_arm_link_resp/elbow_joint/forearm_link_resp/wrist_1_joint/wrist_1_link_resp/wrist_2_joint/wrist_2_link_resp/wrist_3_joint/wrist_3_link_resp/tip')
    simTarget=sim.getObject('/UR5e_dynamic/target')

    # ik env
    ikEnv=simIK.createEnvironment()
    ikGroup=simIK.createIkGroup(ikEnv)
    ikElement, simToIkObjectMapping = simIK.addIkElementFromScene(ikEnv,ikGroup,simBase,simTip,simTarget,simIK.constraint_pose)
    simIK.setIkElementPrecision(ikEnv,ikGroup,ikElement,[0.00005,0.1*np.pi/180])

    #
    ikJointHandles = [simToIkObjectMapping[id] for id in jointHandleList]
    ikTarget=simToIkObjectMapping[simTarget]
    ikBase=simToIkObjectMapping[simBase]


    def setjoint_value(state):
        for i in range(len(state)):
            sim.setJointPosition(jointHandleList[i], state[i])

    def finding_configuration(targetPose):
        simIK.setObjectPose(ikEnv,ikTarget,ikBase, targetPose)
        thresholdDist = 0.3 #0.65
        maxTime = 0.01
        metric = [1,1,1,0.1]
        # thresholdDist: a distance indicating when IK should be computed in order to try to bring the tip onto the target:
        #                since the search algorithm proceeds by generating random configurations, many of them produce a tip pose that is too far from the target pose to run IK successfully.
        #                Choosing a large value will result in slow calculations, choosing a small value might produce a smaller subset of solutions.
        #                Distance between two poses is calculated using a metric (see metric argument below).
        # maxTime: the upper time limit, in seconds, after which the function returns.
        # metric: a table to 4 values indicating a metric used to compute pose-pose distances: distance=sqrt((dx*metric[1])^2+(dy*metric[2])^2+(dz*metric[3])^2+(angle*metric[4])^2).
        state=simIK.findConfig(ikEnv,ikGroup,ikJointHandles,thresholdDist, maxTime, metric)
        if state is not None:
            if state[2] < 0.0:
                return state

    sim.startSimulation()

    try:
        while True:
            targets=sim.getObject('/testTarget5')
            targetPose = sim.getObjectPose(targets, simBase) # pose: the position and quaternion of the object (x,y,z,qx,qy,qz,qw).

            simulationTime = sim.getSimulationTime()
            state = finding_configuration(targetPose)
            if state is not None:
                setjoint_value(state)

    finally:
        sim.stopSimulation()


if __name__=="__main__":
    # solving()
    finding_config()