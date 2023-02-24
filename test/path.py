import os
import pathlib

# get the file path of the current script
file_path = os.path.realpath(__file__)
print(file_path)

#
path = pathlib.Path('./')
print(path)