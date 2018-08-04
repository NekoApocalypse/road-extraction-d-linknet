import cv2
import os
import random
import numpy as np

def read_as_array(filename):
    data_array = cv2.imread(filename)

    return data_array

def random_split_array(img_array, label_array):
    incision_num = 50
    incision_img_w = 1024
    incision_img_h = 1024

    w, h, _ = img_array.shape

    left_top_points_x = random.sample(range(0, w - incision_img_w), incision_num)
    left_top_points_y = random.sample(range(0, h - incision_img_h), incision_num)

    r_img_arrays = []
    r_label_arrays = []

    for i in range(incision_num):
        tmp_img_arr = img_array[left_top_points_x[i]:(left_top_points_x[i]+1024), left_top_points_y[i]:(left_top_points_y[i]+1024), :]
        tmp_label_arr = label_array[left_top_points_x[i]:(left_top_points_x[i]+1024), left_top_points_y[i]:(left_top_points_y[i]+1024), :]

        r_img_arrays.append(tmp_img_arr)
        r_label_arrays.append(tmp_label_arr)

    return r_img_arrays, r_label_arrays

def cross_split_array(img_array):
    stride = 2
    width, height, _ = img_array.shape

    new_w = int(width / stride)
    new_h = int(height / stride)

    r_arrays = []

    for i in range(stride):
        for j in range(stride):
            tif_tmp_arr = img_array[new_w * i:new_w * (i + 1), new_h * j:new_h * (j + 1), :]
            r_arrays.append(tif_tmp_arr)

    return r_arrays

def cross_incision(files):
    for file in files:
        file_name = file.split('.')[0]
        file_format = file.split('.')[-1]
        # print(file_name, file_format)

        if file_format == 'tif':
            tif_array = read_as_array(directory + file)
            bmp_array = read_as_array(directory + file_name + '_gt.bmp')

            new_tif_arrays = cross_split_array(tif_array)
            new_bmp_arrays = cross_split_array(bmp_array)

            print('random split ' + file_name)
            for i in range(len(new_tif_arrays)):
                cv2.imwrite(directory + 'cross/' + file_name + '_' + str(i) + '.tif', new_tif_arrays[i])
                cv2.imwrite(directory + 'cross/' + file_name + '_' + str(i) + '.bmp', new_bmp_arrays[i])

def random_incision(files, directory):
    for file in files:
        file_name = file.split('.')[0]
        file_format = file.split('.')[-1]

        if file_format == 'tif':
            tif_array = read_as_array(directory + file)
            bmp_array = read_as_array(directory + file_name + '_gt.bmp')

            new_tif_arrays, new_bmp_arrays = random_split_array(tif_array, bmp_array)

            print('random split ' + file_name)
            print(len(new_tif_arrays))
            for i in range(len(new_tif_arrays)):
                cv2.imwrite(directory + 'random/' + file_name + '_' + str(i) + '.tif', new_tif_arrays[i])
                cv2.imwrite(directory + 'random/' + file_name + '_' + str(i) + '.bmp', new_bmp_arrays[i])


"""
    mode: how to split the image
    img_array: image data(array format)
"""
def split(files, directory, mode='cross'):
    if mode == 'cross':
        cross_incision(files)
    elif mode == 'random':
        random_incision(files, directory)


if __name__ == '__main__':
    directory = './data/'
    files = os.listdir(directory)
    split(files, directory, mode='random')