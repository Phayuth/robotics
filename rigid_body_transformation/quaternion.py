import numpy as np

# quaternion = q0 + iq1 + jq2 +kq3
# is a rotation by theta about unit vector n = (nx,ny,nz)

def quaternion_from_theta_and_vector(theta,n):
    norm_n = np.sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]) # calculate norm of vector
    nx = n[0]/norm_n # unit vector in x direction
    ny = n[1]/norm_n # unit vector in y direction
    nz = n[2]/norm_n # unit vector in z direction

    q0 = np.cos(theta/2)
    q1 = nx*np.sin(theta/2)
    q2 = ny*np.sin(theta/2)
    q3 = nz*np.sin(theta/2)

    return np.array([q0,q1,q2,q3])

def quaternion_to_theta_and_vector(q):
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    theta = np.arccos(q0)*2
    nx = q1/np.sin(theta/2)
    ny = q2/np.sin(theta/2)
    nz = q3/np.sin(theta/2)

    return theta, np.array([nx,ny,nz])

if __name__ == "__main__":

    unit_vector_x = np.array([1,0,0])
    unit_vector_y = np.array([0,1,0])
    unit_vector_z = np.array([0,0,1])

    theta = np.deg2rad(90)
    print("==>> theta: ", theta)

    q = quaternion_from_theta_and_vector(theta,unit_vector_x)
    print("==>> q: ", q)
    theta_ori,n_ori = quaternion_to_theta_and_vector(q)
    print("==>> theta_ori: ", theta_ori)
    print("==>> n_ori: ", n_ori)