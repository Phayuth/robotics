import sim
import time

print ('Program started')
sim.simxFinish(-1) # just in case, close all opened connections
clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to CoppeliaSim
if clientID!=-1:
    ret, collection = sim.simxGetCollectionHandle(clientID, 'collectionCollision', sim.simx_opmode_blocking)
    print(f"==>> ret: \n{ret}")
    print(f"==>> collection: \n{collection}")
    rr , state = sim.simxCheckCollision(clientID, collection, sim.sim_handle_all, sim.simx_opmode_streaming)
    print(f"==>> rr: \n{rr}")
    print(f"==>> state: \n{state}")
else:
    print ('Failed connecting to remote API server')
print ('Program ended')
