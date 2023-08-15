import yaml
import numpy as np
import json


def read_json(path):
    with open(path, 'r') as s:
        loaded = yaml.safe_load(s)
        dataPoint = loaded['data']
        print(type(dataPoint))
        print(len(dataPoint))


def read_txt_to_numpy(txtFileString):
    txtFile = open(txtFileString, 'r')
    line = txtFile.readlines()
    line = [x.strip("\n[]") for x in line]
    line = [y.split(' ') for y in line]
    array = np.array(line).astype(np.float64)
    return array


def print_dict(dict):
    for sub in dict:
        print(sub, ':', dict[sub])


def write_dict_to_file(dictionary, file_path):
    with open(file_path, 'w') as file:
        json.dump(dictionary, file, indent=4)

    print("Dictionary saved to", file_path)


if __name__ == "__main__":

    perfMatrix = {
        "totalPlanningTime": 0.0,
        "KCDTimeSpend": 0.0,
        "planningTimeOnly": 0.0,
        "numberOfKCD": 0,
        "avgKCDTime": 0.0,
        "numberOfNodeTreeStart": 0,
        "numberOfNodeTreeGoal": 0,
        "numberOfNode": 0,
        "numberOfMaxIteration": 0,
        "numberOfIterationUsed": 0,
        "searchPathTime": 0.0,
        "numberOfPath": 0,
        "numberOfPathPruned": 0}
    print_dict(perfMatrix)


    path = './open3d/depthcloud.yaml'
    read_json(path)

