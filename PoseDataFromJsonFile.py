import h5py as h5
import json
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


def uv_of_joints_saveIn_npy(data_name):
    json_path = os.path.join("JsonFiles", data_name, "*.json")
    save_to_path = os.path.join("extractedData", data_name, "joints.npy")

    # open and sort the data in the path by the datum
    json_files = list(filter(os.path.isfile, glob.glob(json_path)))
    json_files.sort(key=lambda x: os.path.getmtime(x))

    frame_num = len(json_files)

    joints_array = np.zeros((36, frame_num))  # there are 18 joints to be detected (x,y) => 18*2 = 36
    j = 0
    for file in json_files:
        with open(file) as json_file:
            data = json.load(json_file)
            for x in data['people']:
                array = np.asarray(x['pose_keypoints_2d'])
                for i in range(0, 36):
                    joints_array[i, j] = array[i]
        j += 1
    np.save(save_to_path, joints_array)
    print("json files have been saved to ", save_to_path)


if __name__ == '__main__':
    # define folder path
    data_name = "E_von hinten ohne Schulterstütze"
    uv_of_joints_saveIn_npy(data_name)
