import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def world_coordinate_system(data_name):
    depth_data_path = os.path.join("extractedData", data_name, "depth.npy")
    depth = np.load(depth_data_path)
    '''
         u         fx 0  cx     X
        (v) = s * [0  fy cy] * (Y)
         1         0  0   1     Z

         x           fx 0  cx          u
        (y) = 1/s * [0  fy cy]^(-1) * (v)
         Z           0  0   1          1
    '''
    #  undistor funktion needs Umat inputs wich are simalar to np.float32
    cameraMatrix = np.float32([[3.6624647711703801e+02, 0., 2.5395654088270612e+02],
                               [0., 3.6583977629830412e+02, 2.0632120087909621e+02],
                               [0., 0., 1.]])
    distCoeff = np.float32([9.8409165280489860e-02, -2.7659785802806891e-01, -1.2205135326263088e-03,
                            -9.1259351365805116e-04, 8.6288563966411561e-02])

    registered = 'registered'
    sz = (424, 512)  # sz kinect 2.0
    s = 1  # because of linearity
    # cameraMatrixK_inv = np.linalg.inv(cameraMatrixK)
    cameraMatrix_inv = np.linalg.inv(cameraMatrix)

    coordinates = np.dstack((np.repeat((np.ones((1, sz[0])) * np.arange(sz[0])).T, sz[1], axis=1),
                             np.repeat((np.ones((1, sz[1])) * np.arange(sz[1])), sz[0], axis=0),
                             np.ones(sz)))

    # xyzK = np.transpose(1/s * np.dot(cameraMatrixK_inv, np.transpose(coordinateK, (0, 2, 1))), (1, 2, 0))
    xyz = np.transpose(1 / s * np.dot(cameraMatrix_inv, np.transpose(coordinates, (0, 2, 1))), (1, 2, 0))

    #### undistort the image
    # xyzK = cv2.undistort(xyzK[:,:,:2], cameraMatrixK, distCoeffK)  # undistort image
    # xyzK = np.dstack((xyzK, np.ones(szK)))  # calculate Homogeneous
    xyz = cv2.undistort(xyz[:, :, :2], cameraMatrix, distCoeff)
    xyz = np.dstack((xyz, np.ones(sz)))

    #### Normalization of the image X_norm = X / |X|
    # magnitudeK = np.sum(xyzK**2, axis=2)**0.5
    magnitude = np.sum(xyz ** 2, axis=2) ** 0.5
    # xyzK = xyzK / np.dstack((magnitudeK, magnitudeK, magnitudeK))
    xyz = xyz / np.dstack((magnitude, magnitude, magnitude))

    # .....
    data_size = np.shape(depth)
    depth_data_all_frames = np.zeros((424, 512, 3, data_size[2]))
    for i in tqdm(range(0, data_size[2]), desc="get world coordinate of the point cloud"):
        XYZ = xyz * np.dstack((depth[:, :, i],
                               depth[:, :, i],
                               depth[:, :, i]))
        depth_data_all_frames[:, :, :, i] = XYZ

    save_to_path = os.path.join("extractedData", data_name, "world_coordinate.npy")
    np.save(save_to_path, depth_data_all_frames)
    print("world_coordinate.npy have ben saved")


if __name__ == "__main__":
    data_name = "E_von hinten ohne Schulterstütze"
    world_coordinate_system(data_name)
