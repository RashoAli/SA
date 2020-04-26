import os
import numpy as np
from numpy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


def adjusting_plane(data, pixel_position, kernel_size, vector_length):
    frame_number = 0
    kernel = data[frame_number, pixel_position[0] - kernel_size:pixel_position[0] + kernel_size + 1,
             pixel_position[1] - kernel_size:pixel_position[1] + kernel_size + 1]

    # 1: center of gravity
    size = np.shape(kernel)
    Zc = int(1 / (size[0] * size[1]) * np.sum(kernel))

    # 2: massen Matrix
    x_ones_array = np.ones(size)
    x_repeat_array = np.arange(size[0])  # np.arange(size[1]+1)-Yc
    for i in range(0, size[0]):
        x_ones_array[i, :] = np.multiply(x_ones_array[i, :], x_repeat_array)

    X_kernel = x_ones_array - int(size[0] / 2)
    Y_kernel = X_kernel.transpose()
    Z_kernel = kernel - Zc

    xx = np.sum(np.multiply(X_kernel, X_kernel))
    xy = np.sum(np.multiply(X_kernel, Y_kernel))
    xz = np.sum(np.multiply(X_kernel, Z_kernel))
    yy = np.sum(np.multiply(Y_kernel, Y_kernel))
    zy = np.sum(np.multiply(Y_kernel, Z_kernel))
    zz = np.sum(np.multiply(Z_kernel, Z_kernel))

    M = np.array([[xx, xy, xz], [xy, yy, zy], [xz, zy, zz]])
    w, v = LA.eig(M)

    # the eigenVector with the smalest eigenValue is the tangent vector
    xs = X_kernel.flatten()
    ys = Y_kernel.flatten()
    zs = Z_kernel.flatten() * kernel_size / np.max(Z_kernel)

    #  plot tangent surface
    midele_point_index = int((np.shape(zs)[0] - 1) / 2)
    xs1 = v[0][0] * vector_length
    ys1 = v[0][1] * vector_length
    zs1 = zs[midele_point_index] + v[0][2] * vector_length

    return xs1, ys1, zs1, xs, ys, zs  # xs, ys, zs:all points , xs1, ys1, zs1 just the joint point


def plot_the_data(xs1, ys1, zs1, xs, ys, zs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, marker='.')

    #  plot tangent surface
    ax.scatter(xs1, ys1, zs1, marker='^', color="r")

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def FilterPoseData(data_name):
    npy_dir = os.path.join("extractedData", data_name, "ir.npy")
    data = np.load(npy_dir)
    pixel_position = [193, 329]
    kernel_size = 10
    xs_temp = []
    ys_temp = []
    zs_temp = []

    vector_length = 40
    for i in range(5, 15):
        xs1, ys1, zs1, xs, ys, zs = adjusting_plane(data, pixel_position, i, vector_length)
        xs_temp.append(xs1)
        ys_temp.append(ys1)
        zs_temp.append(zs1)
    xs1 = np.asarray(xs_temp)
    ys1 = np.asarray(ys_temp)
    zs1 = np.asarray(zs_temp)
    plot_the_data(xs1, ys1, zs1, xs, ys, zs)
    # plt.imshow(data[0,:,:])
    # plt.show()


def temp_function_for_XZYJoints(data_name, max_depth):
    point_cloud_path = os.path.join("extractedData", data_name, "world_coordinate.npy")
    joints_path = os.path.join("extractedData", data_name, "joints.npy")
    data = np.load(point_cloud_path)
    joints_position = np.load(joints_path)
    # FilterPoseData(data_name)
    frame_number = 14
    data_frame_14 = data[:, :, :, frame_number]
    human_depth_data = np.where(data_frame_14[:, :, 2].flatten() > max_depth)
    flatten_xs = np.delete(data_frame_14[:, :, 0], human_depth_data[0]).flatten()
    flatten_ys = np.delete(data_frame_14[:, :, 1], human_depth_data[0]).flatten()
    flatten_zs = np.delete(data_frame_14[:, :, 2], human_depth_data[0]).flatten()

    elements_number = len(flatten_zs)
    each_element = 50
    remove_ellements = np.delete(np.arange(0, elements_number), np.arange(0, elements_number, each_element))

    xs = np.delete(flatten_xs, remove_ellements)
    ys = np.delete(flatten_ys, remove_ellements)
    zs = np.delete(flatten_zs, remove_ellements)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, marker='.')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def get_joints_XYZ(data_name):
    point_cloud_path = os.path.join("extractedData", data_name, "world_coordinate.npy")
    joints_path = os.path.join("extractedData", data_name, "joints.npy")
    data = np.load(point_cloud_path)
    joints_position = np.load(joints_path)

    joints_xyz = np.zeros([3, 18, 50])
    for frame_number in range(0, 50):
        data_frame = data[:, :, :, frame_number]
        for i in range(0, 18):
            joint_postion = np.array(
                [int(joints_position[2 * i, frame_number]), int(joints_position[2 * i + 1, frame_number])])
            joints_xyz[0, i, frame_number] = data_frame[joint_postion[0], joint_postion[1], 0]
            joints_xyz[1, i, frame_number] = data_frame[joint_postion[0], joint_postion[1], 1]
            joints_xyz[2, i, frame_number] = data_frame[joint_postion[0], joint_postion[1], 2]

    save_to_path = os.path.join("extractedData", data_name, "joints_XYZ.npy")
    np.save(save_to_path, joints_xyz)


if __name__ == '__main__':
    data_name = "E_von hinten ohne Schulterstütze"
    get_joints_XYZ(data_name)
