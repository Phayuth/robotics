import os
import pathlib

# get the file path of the current script
file_path = os.path.realpath(__file__)
print(file_path)

#
path = pathlib.Path('./')
print(path)


# add path so python script can back up one level and go to another different dir to find another file
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))