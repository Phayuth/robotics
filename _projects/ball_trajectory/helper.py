import numpy as np

def read_txt_to_numpy(txtFileString):
    txtFile = open(txtFileString, 'r')
    line = txtFile.readlines()
    line = [x.strip("\n[]") for x in line]
    line = [y.split(' ') for y in line]
    array = np.array(line).astype(np.float64)
    return array