def Reshape_map(map):
    r_map = []
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            for k in range(map.shape[2]):
                r_map.append([i, j, k, map[i,j,k]])
    return r_map