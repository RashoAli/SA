from openHDF import ExtractDataFrom_hdf
from PoseDataFromJsonFile import uv_of_joints_saveIn_npy
from xyd2XYZ import world_coordinate_system
from Prepare_npy_file_to_Blender import get_joints_XYZ

import os
import numpy as np
import matplotlib.pyplot as plt

data_name = "E_von hinten ohne Schulterstütze"  # video name

#  Extract data from hdf to extractedData folder
ExtractDataFrom_hdf(data_name)

# get the json files
uv_of_joints_saveIn_npy(data_name)

# get the world point cloud
world_coordinate_system(data_name)

# filter the data and Prepare for Blender
get_joints_XYZ(data_name)
jointsData = np.load(os.path.join("extractedData", data_name, "joints_XYZ.npy"))
