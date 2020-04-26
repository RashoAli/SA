
'''
    Analysis of cuboid
    @author: fehlandt
    @date: 05.09.2018
'''

import numpy as np
import pandas as pd
import os
import cv2
import glob
from skimage.morphology import erosion, dilation

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy

import sys

sys.path.insert(0, r'../../preprocessing')
from otsu import otsuthreshold
from display_hdf import rgb2gray, read_h5py
plot = True


def undistort_unproject_pts(pts_uv, camera_matrix, dist_coefs):
    """
    This function converts a set of 2D image coordinates to vectors in pinhole camera space.
    Hereby the intrinsics of the camera are taken into account.
    UV is converted to normalized image space (think frustum with image plane at z=1) then undistored
    adding a z_coordinate of 1 yield vectors pointing from 0,0,0 to the undistored image pixel.
    @return: ndarray with shape=(n, 3)

    """
    pts_uv = np.array(pts_uv)
    num_pts = pts_uv.size / 2

    if pts_uv.shape[1] == 2:
        pass
    elif pts_uv.shape[0] == 2:
        pts_uv = pts_uv.T
    else:
        raise('change shape of pointcloud to (num_points, 2) oder (2,num_points)')


    pts_uv.shape = (int(num_pts), 1, 2)  # pts_uv = pts_uv.reshape(int(num_pts), 1, 2)

    pts_uv = cv2.undistortPoints(pts_uv, camera_matrix, dist_coefs)
    pts_3d = cv2.convertPointsToHomogeneous(np.float32(pts_uv))
    pts_3d.shape = (int(num_pts), 3)
    return pts_3d


#### Read Data
camera = 'kinect2'; print('\n\t' + camera + '\n\t_______\n')

# calibration data
if camera == 'd435':
    cameraMatrix = [[704.1921647187825, 0.0,               639.0229471467588],
                    [0.0,               623.5973800815007, 407.3622090168924],
                    [0.0,               0.0,               1.0              ]]
    distCoeff = [0.012750446981508973, -0.031207559401856103, 0.01865365324504344,
                 0.0053737731098355765, 0.08069041049740713]
    registered = 'color_aligned'
    sz = (720, 1280)  # sz D435
elif camera == 'kinect2':
    cameraMatrix = [[3.6624647711703801e+02, 0.,                     2.5395654088270612e+02],
                    [0.,                     3.6583977629830412e+02, 2.0632120087909621e+02],
                    [0.,                     0.,                      1.                   ]]
    distCoeff = [9.8409165280489860e-02, -2.7659785802806891e-01, -1.2205135326263088e-03,
                 -9.1259351365805116e-04, 8.6288563966411561e-02]
    registered = 'registered'
    sz = (424, 512)  # sz kinect 2.0
else:
    raise('wrong camera string')
cameraMatrix = np.array(cameraMatrix)
distCoeff = np.array(distCoeff)
directory = 'box_' + camera

# initialization
frame_number = len(glob.glob(os.path.join(directory, '1_*.hdf')))

depth_stack = np.zeros((sz[0], sz[1], frame_number))
depth_threshold = 1500
kernel = [[0, 1, 0],
          [1, 1, 1],
          [0, 1, 0]]

test = False
if test:
    end = 2
else:
    end = frame_number
for frame in range(end):
    depth = read_h5py(os.path.join(directory, '1_' + str(frame) + '.hdf'), list='SensorData', sublist='depth')
    color = read_h5py(os.path.join(directory, '1_' + str(frame) + '.hdf'), list='SensorData', sublist=registered)
    print('\tread frame ' + str(frame))

    # preprocessing of depth frame
    mask = np.logical_and(0 < depth, depth < depth_threshold)
    mask = dilation(erosion(mask, kernel))
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    gray *= mask
    threshold = otsuthreshold(gray[gray!=0])
    mask = np.logical_and(0 < gray, gray < threshold)
    depth *= mask
    color[:, :, 0] *= mask
    color[:, :, 1] *= mask
    color[:, :, 2] *= mask

    depth_stack[:,:,frame] = depth

    if plot:
        cv2.imshow('color', color)
        depth_display = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow('depth', depth_display)
        cv2.waitKey(0)

# convert image (u_i, v_j) into camera coordinate system (X_ij, Y_ij, Z_ij) with camera matrix and scaling factor
'''
     u         fx 0  cx     X
    (v) = s * [0  fy cy] * (Y)
     1         0  0   1     Z

     x           fx 0  cx          u
    (y) = 1/s * [0  fy cy]^(-1) * (v)
     Z           0  0   1          1
'''
s = 1  # because of linearity
# cameraMatrixK_inv = np.linalg.inv(cameraMatrixK)
cameraMatrix_inv = np.linalg.inv(cameraMatrix)

coordinates = np.dstack(( np.repeat((np.ones((1, sz[0])) * np.arange(sz[0])).T, sz[1], axis=1),
                          np.repeat((np.ones((1, sz[1])) * np.arange(sz[1]))  , sz[0], axis=0),
                          np.ones(sz) ))

# xyzK = np.transpose(1/s * np.dot(cameraMatrixK_inv, np.transpose(coordinateK, (0, 2, 1))), (1, 2, 0))
xyz = np.transpose(1/s * np.dot(cameraMatrix_inv, np.transpose(coordinates, (0, 2, 1))), (1, 2, 0))

#### undistort the image
# xyzK = cv2.undistort(xyzK[:,:,:2], cameraMatrixK, distCoeffK)  # undistort image
# xyzK = np.dstack((xyzK, np.ones(szK)))  # calculate Homogeneous
xyz = cv2.undistort(xyz[:,:,:2], cameraMatrix, distCoeff)
xyz = np.dstack((xyz, np.ones(sz)))

#### Normalization of the image X_norm = X / |X|
# magnitudeK = np.sum(xyzK**2, axis=2)**0.5
magnitude = np.sum(xyz**2, axis=2)**0.5
# xyzK = xyzK / np.dstack((magnitudeK, magnitudeK, magnitudeK))
xyz = xyz / np.dstack((magnitude, magnitude, magnitude))

if plot:
    # display magnitude
    # disp_magnitudeK = deepcopy(magnitudeK)-np.min(np.min(magnitudeK))
    # disp_magnitudeK = disp_magnitudeK/np.max(np.max(disp_magnitudeK))
    disp_magnitude = deepcopy(magnitude-np.min(np.min(magnitude)))
    disp_magnitude = disp_magnitude/np.max(np.max(disp_magnitude))

    # cv2.imshow(camera[0], disp_magnitudeK)
    # cv2.imshow(camera[1], disp_magnitudeD)
    cv2.imshow(camera, disp_magnitude)
    cv2.waitKey(0)

#### create and save pointcloud: project normalized vector into 3d space
plot = True
print('\nsave pointclouds\n')
# plot = False
dir_store = os.path.join(directory, 'pointclouds/')
for frame in range(end):
    print('\tpointcloud ' + str(frame))
    # XYZ_K = xyzK * np.dstack(( depthK[:, :, frame],
    #                            depthK[:, :, frame],
    #                            depthK[:, :, frame] ))
    XYZ = xyz * np.dstack(( depth_stack[:, :, frame],
                               depth_stack[:, :, frame],
                               depth_stack[:, :, frame] ))

    # nonZero = np.where(np.sum(XYZ_K, axis=2) != 0)
    # XYZ_K = XYZ_K[nonZero[0], nonZero[1]]
    nonZero = np.where(np.sum(XYZ, axis=2) != 0)
    XYZ = XYZ[nonZero[0], nonZero[1]]

    if plot:
        #if i == 0:
        colorRand = np.random.random((3, 1))  # choose color
        cK = np.array([colorRand[0, 0], colorRand[1, 0], colorRand[2, 0]])
        cD = (cK + 0.5) % 1  # takes opposite color, because 0 < c < 1
        marker = '.'

        # create a 3D subplot
        fig = plt.figure(1)
        plt.show()
        sub3D = fig.add_subplot(111, projection='3d')
        # sub3D.plot(XYZ_K[:, 0], XYZ_K[:, 1], XYZ_K[:, 2], marker, c=cK, label='Kinect 2.0')
        sub3D.plot(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2], marker, c=cD, label='Realsense D435')
        sub3D.legend()
        plt.show()

    # write pointcloud into *.csv file
    # df_XYZ_K = pd.DataFrame(XYZ_K)
    # df_XYZ_K.to_csv(dir_store + 'pointcloud_' + 'kinect2' + '_frame' + str(frame) + '.csv')
    df_XYZ = pd.DataFrame(XYZ)
    df_XYZ.to_csv(dir_store + 'pointcloud_' + camera + '_frame' + str(frame) + '.csv')

    # write pointcloud into *.txt file
    # fileK = open(dir_store + 'pointcloud_' + 'kinect2' + '_frame' + str(frame) + '.txt', 'w')
    file = open(dir_store + 'pointcloud_' + camera + '_frame' + str(frame) + '.txt', 'w')
    # if XYZ_K.shape[0] > XYZ_D.shape[0]:
    #     end = XYZ_K.shape[0]
    # else:
    #     end =XYZ_D.shape[0]
    end = XYZ.shape[0]
    for points in range(end):
        # if points < XYZ_K.shape[0]:
        #     fileK.write(str(XYZ_K[points, 0]) + ', ' + str(XYZ_K[points, 1]) + ', ' + str(XYZ_K[points, 2]) + ', \n')
        if points < XYZ.shape[0]:
            file.write(str(XYZ[points, 0]) + ', ' + str(XYZ[points, 1]) + ', ' + str(XYZ[points, 2]) + ', \n')
    # fileK.close()
    file.close()

### create CAD model of the cuboid
# cuboid's dimensions/mm:
length = 52.1
width = 39.8
height = 112.0


pass  # TODO: create CAD model for cuboid
