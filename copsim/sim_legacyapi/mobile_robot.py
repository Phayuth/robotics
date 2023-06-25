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


# Pioneer Mobile Robot
errorCode, floor = sim.simxGetObjectHandle(clientID, '/Floor', sim.simx_opmode_oneshot_wait)
errorCode, robot = sim.simxGetObjectHandle(clientID, '/PioneerP3DX', sim.simx_opmode_oneshot_wait)

errorCode, leftMotorHandle = sim.simxGetObjectHandle(clientID, '/PioneerP3DX/leftMotor', sim.simx_opmode_oneshot_wait)
errorCode, rightMotorHandle = sim.simxGetObjectHandle(clientID, '/PioneerP3DX/rightMotor', sim.simx_opmode_oneshot_wait)

errorCode = sim.simxSetJointTargetVelocity(clientID, leftMotorHandle, -0.4, sim.simx_opmode_oneshot_wait)
errorCode = sim.simxSetJointTargetVelocity(clientID, rightMotorHandle, 0.4, sim.simx_opmode_oneshot_wait)

while True:
    pose = sim.simxGetObjectPosition(clientID,robot,sim.handle_world,sim.simx_opmode_streaming)
    print(f"==>> pose: \n{pose}")
