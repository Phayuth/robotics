def extract_path_class_6d(path):

    path_x= []
    path_y= []
    path_z= []
    path_p= []
    path_q= []
    path_r= []

    for i in path:
        x = i.x
        y = i.y
        z = i.z
        p = i.p
        q = i.q
        r = i.r
        path_x.append(x)
        path_y.append(y)
        path_z.append(z)
        path_p.append(p)
        path_q.append(q)
        path_r.append(r)

    return path_x, path_y, path_z, path_p, path_q, path_r

def extract_path_class_3d(path):

    path_x= []
    path_y= []
    path_z= []

    for i in path:
        x = i.x
        y = i.y
        z = i.z
        path_x.append(x)
        path_y.append(y)
        path_z.append(z)

    return path_x, path_y, path_z

def extract_path_class_2d(path):

    path_x= []
    path_y= []

    for i in path:
        x = i.x
        y = i.y
        path_x.append(x)
        path_y.append(y)

    return path_x, path_y