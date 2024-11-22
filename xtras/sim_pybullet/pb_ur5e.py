import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import pybullet as p
import pybullet_data

# connect
clid = p.connect(p.SHARED_MEMORY)
if clid < 0:
    p.connect(p.GUI)
    # p.connect(p.DIRECT)  # for non-graphical version
    # p.connect(p.SHARED_MEMORY_GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# load urdf
p.loadURDF("plane.urdf", [0, 0, 0])
ur5eID = p.loadURDF("./datasave/urdf/ur5e_extract_calibrated.urdf", [0, 0, 0], useFixedBase=True)

# get joint number
numJoints = p.getNumJoints(ur5eID)
print(f"> numJoints: {numJoints}")

# get joint info and link info
jointNames = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
jointIDs = [1, 2, 3, 4, 5, 6]
for i in range(numJoints):
    stateJoints = p.getJointInfo(ur5eID, i)
    # (6, b'wrist_3_joint', 0, 12, 11, 1, 0.0, 0.0, -6.283185307179586, 6.283185307179586, 28.0, 3.141592653589793, b'wrist_3_link', (0.0, 0.0, 1.0), (1.877592835709357e-05, 0.09973050279255603, 7.91315518559599e-06), (0.7070787278000841, -1.1307468292927083e-07, 1.1308365515291548e-07, 0.7071348334600617), 5)
    # jointIndex, jointName, jointType, qIndex, uIndex, flags, jointDamping, jointFriction,  jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, linkName, jointAxis,  parentFramePos,  parentFrameOrn, parentIndex
    link_name = stateJoints[12].decode("utf-8")
    link_id = stateJoints[0]  # Link ID
    print(f"Link {link_name}: ID = {link_id}")

    joint_name = stateJoints[1].decode("utf-8")
    joint_id = stateJoints[0]  # Joint ID
    print(f"Joint {joint_name}: ID = {joint_id}")


def control_single_motor():
    while True:
        jd = 3.14
        p.setJointMotorControl2(bodyIndex=ur5eID, jointIndex=1, controlMode=p.POSITION_CONTROL, targetPosition=jd, targetVelocity=0, positionGain=0.03)
        p.stepSimulation()


def control_array_motors():
    while True:
        jds = [3.14, 0, -1.23, 0, 0, 0]
        p.setJointMotorControlArray(bodyIndex=ur5eID, jointIndices=jointIDs, controlMode=p.POSITION_CONTROL, targetPositions=jds, targetVelocities=[0, 0, 0, 0, 0, 0], positionGains=[0.03, 0.03, 0.03, 0.03, 0.03, 0.03])
        p.stepSimulation()


def get_single_joint_state():
    js = p.getJointState(ur5eID, jointIndex=1)
    print(f"> js: {js}")
    # jointPosition, jointVelocity, jointReactionForces, appliedJointMotorTorque


def get_array_joint_state():
    js = p.getJointStates(ur5eID, jointIndices=jointIDs)
    print(f"> js: {js}")


gripperlinkid = 9
linkstate = p.getLinkState(ur5eID, gripperlinkid, computeForwardKinematics=True)
position, orientation = linkstate[0], linkstate[1]
print(f"> position: {position}")
print(f"> orientation: {orientation}")

while True:
    p.stepSimulation()


p.disconnect()





# https://github.com/josepdaniel/ur5-bullet/blob/master/UR5/UR5Sim.py#L59