import sim
import time
import sys

print("Program Started")
sim.simxFinish(-1)
clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)

if(clientID != -1):
    print('Connected Successfully.')
else:
    sys.exit('Failed To connect.')

time.sleep(1)

print('Setting Up Handles')

# UR5
_, joint1Handle = sim.simxGetObjectHandle(clientID, '/UR5/joint', sim.simx_opmode_oneshot_wait)
_, joint2Handle = sim.simxGetObjectHandle(clientID, '/UR5/link/joint', sim.simx_opmode_oneshot_wait)
_, joint3Handle = sim.simxGetObjectHandle(clientID, '/UR5/link/joint/link/joint', sim.simx_opmode_oneshot_wait)
_, joint4Handle = sim.simxGetObjectHandle(clientID, '/UR5/link/joint/link/joint/link/joint', sim.simx_opmode_oneshot_wait)
_, joint5Handle = sim.simxGetObjectHandle(clientID, '/UR5/link/joint/link/joint/link/joint/link/joint', sim.simx_opmode_oneshot_wait)
_, joint6Handle = sim.simxGetObjectHandle(clientID, '/UR5/link/joint/link/joint/link/joint/link/joint/link/joint', sim.simx_opmode_oneshot_wait)

_, robot = sim.simxGetObjectHandle(clientID, '/UR5', sim.simx_opmode_oneshot_wait)
_, collection  = sim.simxGetCollectionHandle(clientID, 'robotCollection', sim.simx_opmode_oneshot_wait)

# Actuate
# sim.simxSetJointTargetPosition(clientID, joint1Handle, -1.57, sim.simx_opmode_oneshot_wait)
# sim.simxSetJointTargetPosition(clientID, joint2Handle, -1, sim.simx_opmode_oneshot_wait)
# sim.simxSetJointTargetPosition(clientID, joint3Handle, -1.57, sim.simx_opmode_oneshot_wait)
# sim.simxSetJointTargetPosition(clientID, joint4Handle, 1.57, sim.simx_opmode_oneshot_wait)
# sim.simxSetJointTargetPosition(clientID, joint5Handle, 1.57, sim.simx_opmode_oneshot_wait)
# sim.simxSetJointTargetPosition(clientID, joint6Handle, 0, sim.simx_opmode_oneshot_wait)

resp, state = sim.simxCheckCollision(clientID, collection, sim.sim_handle_all, sim.simx_opmode_streaming)

i = 0
while True:
    sim.simxSetJointTargetPosition(clientID, joint1Handle, i, sim.simx_opmode_streaming)
    # sim.simxSetJointPosition(clientID, joint1Handle, i, sim.simx_opmode_streaming)
    i += 0.1
    resp, state = sim.simxCheckCollision(clientID, robot, sim.sim_handle_all, sim.simx_opmode_buffer)
    print(f"==>> state: \n{state}")



# Pioneer Mobile Robot
# errorCode, floor = sim.simxGetObjectHandle(clientID, '/Floor', sim.simx_opmode_oneshot_wait)
# errorCode, robot = sim.simxGetObjectHandle(clientID, '/PioneerP3DX', sim.simx_opmode_oneshot_wait)

# errorCode, leftMotorHandle = sim.simxGetObjectHandle(clientID, '/PioneerP3DX/leftMotor', sim.simx_opmode_oneshot_wait)
# errorCode, rightMotorHandle = sim.simxGetObjectHandle(clientID, '/PioneerP3DX/rightMotor', sim.simx_opmode_oneshot_wait)

# errorCode = sim.simxSetJointTargetVelocity(clientID, leftMotorHandle, -0.4, sim.simx_opmode_oneshot_wait)
# errorCode = sim.simxSetJointTargetVelocity(clientID, rightMotorHandle, 0.4, sim.simx_opmode_oneshot_wait)

# while True:
#     pose = sim.simxGetObjectPosition(clientID,robot,sim.handle_world,sim.simx_opmode_streaming)
#     print(f"==>> pose: \n{pose}")
