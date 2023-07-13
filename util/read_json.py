import yaml

path = './open3d/depthcloud.yaml'

with open(path,'r') as s:
    loaded = yaml.safe_load(s)
    data_point = loaded['data']
    print(type(data_point))
    print(len(data_point))