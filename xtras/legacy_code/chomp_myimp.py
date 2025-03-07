import numpy as np
np.set_printoptions(linewidth=2000)

n = 10 # nwaypoints
time_interval = 0.1


diff_rule_length = 7
# diff_rule = [0, 0, -1, 1, 0, 0, 0]
diff_rule = [0, 0, 1, -2, 1, 0, 0]
half_length = diff_rule_length // 2
diff_matrix = np.zeros([n + 1, n])
for i in range(0, n + 1):
    for j in range(-half_length, half_length):
        index = i + j
        if index >= 0 and index < n:
            diff_matrix[i, index] = diff_rule[j + half_length]
diff_matrix = diff_matrix / (time_interval**2)
print(f"> diff_matrix: \n{diff_matrix}")


A = diff_matrix.T.dot(diff_matrix)
print(f"> A: \n{A}")
print(f"> A.shape: {A.shape}")
Ainv = np.linalg.inv(A)


d = 6  # 6 joints
start = np.random.uniform(-np.pi, np.pi, size=(1, d))
end = np.random.uniform(-np.pi, np.pi, size=(1, d))
xi = np.linspace(start.flatten(), end.flatten(), n, endpoint=True)
print(f"> xi: {xi}")
print(f"> xi.shape: {xi.shape}")
# xi = np.random.uniform(-np.pi, np.pi, size=(n, d))
# print(f"> xi.shape: {xi.shape}")


link_smooth_weight = np.array([1.0]*d)[None]  # add additional dimension from (9,) to (1,9)
print(f"> link_smooth_weight: {link_smooth_weight}")
print(f"> link_smooth_weight.shape: {link_smooth_weight.shape}")

ed = np.zeros([xi.shape[0] + 1, xi.shape[1]])
ed[0] = diff_rule[diff_rule_length // 2 - 1] * start / time_interval
# if not goal_set_proj:
#     ed[-1] = diff_rule[diff_rule_length // 2] * end / time_interval
print(f"> ed: {ed}")
print(f"> ed.shape: {ed.shape}")

velocity = diff_matrix.dot(xi)
print(f"> velocity: {velocity}")
print(f"> velocity.shape: {velocity.shape}")

velocity_norm = np.linalg.norm((velocity + ed) * link_smooth_weight, axis=1)
print(f"> velocity_norm: {velocity_norm}")
print(f"> velocity_norm.shape: {velocity_norm.shape}")

smoothness_loss = 0.5 * velocity_norm**2
print(f"> smoothness_loss: {smoothness_loss}")
print(f"> smoothness_loss.shape: {smoothness_loss.shape}")

smoothness_grad = A.dot(xi) + diff_matrix.T.dot(ed)
smoothness_grad *= link_smooth_weight
print(f"> smoothness_grad: {smoothness_grad}")
print(f"> smoothness_grad.shape: {smoothness_grad.shape}")

smoothness_loss_sum = smoothness_loss.sum()
print(f"> smoothness_loss_sum: {smoothness_loss_sum}")
print(f"> smoothness_loss_sum.shape: {smoothness_loss_sum.shape}")

smoothness_base_weight = 0.1  # 0.1 weight for smoothness cost in total cost
cost_schedule_boost = 1.02  # cost schedule boost for smoothness cost weight

step = 1  # integer increasing by 1 everytime we update
smoothness_weight = smoothness_base_weight * cost_schedule_boost**step
print(f"> smoothness_weight: {smoothness_weight}")

weighted_smooth = smoothness_weight * smoothness_loss_sum
print(f"> weighted_smooth: {weighted_smooth}")
print(f"> weighted_smooth.shape: {weighted_smooth.shape}")