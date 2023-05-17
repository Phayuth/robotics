import numpy as np

# GENERATING DATA -------------------------------------------------------------------------------------------------------------
class NLinkArm(object):
 
    def __init__(self, link_lengths, joint_angles):
        self.n_links = len(link_lengths)
        if self.n_links != len(joint_angles):
            raise ValueError()

        self.link_lengths = np.array(link_lengths)
        self.joint_angles = np.array(joint_angles)
        self.points = [[0, 0] for _ in range(self.n_links + 1)]

        self.lim = sum(link_lengths)
        self.update_points()

    def update_joints(self, joint_angles):
        self.joint_angles = joint_angles
        self.update_points()

    def update_points(self):
        for i in range(1, self.n_links + 1):
            self.points[i][0] = self.points[i - 1][0] + self.link_lengths[i - 1] * np.cos(np.sum(self.joint_angles[:i]))
            self.points[i][1] = self.points[i - 1][1] + self.link_lengths[i - 1] * np.sin(np.sum(self.joint_angles[:i]))

        self.end_effector = np.array(self.points[self.n_links]).T


def detect_collision(line_seg, circle):
    a_vec = np.array([line_seg[0][0], line_seg[0][1]])
    b_vec = np.array([line_seg[1][0], line_seg[1][1]])
    c_vec = np.array([circle[0], circle[1]])
    radius = circle[2]
    line_vec = b_vec - a_vec
    line_mag = np.linalg.norm(line_vec)
    circle_vec = c_vec - a_vec
    proj = circle_vec.dot(line_vec / line_mag)
    if proj <= 0:
        closest_point = a_vec
    elif proj >= line_mag:
        closest_point = b_vec
    else:
        closest_point = a_vec + line_vec * proj / line_mag
    if np.linalg.norm(closest_point - c_vec) > radius:
        return False

    return True


def get_occupancy_grid(arm, obstacles):

    grid = [[0 for _ in range(M)] for _ in range(M)]
    theta_list = [2 * i * np.pi / M for i in range(-M // 2, M // 2 + 1)]
   
    dataset = []

    for i in range(M):
        for j in range(M):
            arm.update_joints([theta_list[i], theta_list[j]])
            points = arm.points
            collision_detected = False
            for k in range(len(points) - 1):
                for obstacle in obstacles:
                    line_seg = [points[k], points[k + 1]]
                    collision_detected = detect_collision(line_seg, obstacle)
                    if collision_detected:
                        break
                if collision_detected:
                    break
            grid[i][j] = int(collision_detected)

            if int(collision_detected) == 1:
                collision_stat = 1
            elif int(collision_detected) == 0:
                collision_stat = -1
            dataset.append([theta_list[i],theta_list[j],collision_stat])

    return np.array(grid), dataset

# Simulation parameters
M = 10 # number of sample to divide into and number of grid cell
obstacles = [[1.75, 0.75, 0.6], [0.55, 1.5, 0.5], [0, -1, 0.25], [-1.5, -1.5, 0.25]] # x y radius
arm = NLinkArm([1, 1], [0, 0])
grid, dataset = get_occupancy_grid(arm, obstacles)

















# dataset
dataset = np.array(dataset)
data = dataset[:,0:2]
y = dataset[:,2]

def fx(queryPoint):
    term = []
    for ind, xi in enumerate(data):
        norm = np.linalg.norm([(xi[0] - queryPoint[0]),(xi[1] - queryPoint[1])])
        K = np.exp(-g*(norm**2))
        term.append(alpha[ind] * K)
    ypred = np.sign(sum(term))
    return ypred

# Fastron
N = data.shape[0]        # number of datapoint = number of row the dataset has
d = data.shape[1]        # number of dimensionality = number of columns the dataset has (x1, x2, ..., xn)
g = 10                   # kernel width
beta = 100               # conditional bias
maxUpdate = 100        # max update iteration
maxSupportPoints = 1500  # max support points
G = data @ data.T                                 # gram matrix of dataset
alpha = np.zeros(N)                               # weight, init at zero
F = np.array([fx(data[i]) for i in range(N)])     # hypothesis

# active learning parameters
allowance = 800          # number of new samples
kNS = 4                  # number of points near supports
sigma = 0.5              # Gaussian sampling std
exploitP = 0.5           # proportion of exploitation samples



def original_kernel_update():
    """Brute force update, unneccessary calculation"""
    for iter in range(maxUpdate):
        print(iter)
        for i in range(N):
            margin = y[i] * fx(data[i])
            if margin <= 0:
                alpha[i] += y[i]




# def fastron_model_update():
#     for iter in range(maxUpdate):
#         for i in range(N):
#             print(f"at i = {i}")
#             print(f"yi = {y[i]} , Fi = {F[i]} , alphai = {alpha[i]}")
#             if margin:=y[i]*fx(data[i] - alpha[i]) > 0 and alpha != 0.0:
#                 j = np.argmax(margin)
#                 print(j)




queryPoint = np.array([1,1]) # queryPoint[0] = q1, queryPoint[1] = q2
qP = fx(queryPoint)
print(f"==>> qP: \n{qP}")

original_kernel_update()
print(alpha)

qP = fx(queryPoint)
print(f"==>> qP: \n{qP}")

# fastron_model_update()
print("end")