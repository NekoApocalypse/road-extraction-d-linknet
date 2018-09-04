import random
import imageio
import numpy as np
import glob
import os
from skimage import transform


def random_rotation(image_arr, msk_arr):
    # pick a random degree
    random_degree = random.uniform(-25, 25)
    rotated_img = transform.rotate(image_arr, random_degree)
    rotated_msk = transform.rotate(msk_arr, random_degree)

    rotated_img = rotated_img * 255.0
    rotated_img = rotated_img.astype(np.int8)

    return rotated_img, rotated_msk


def vertical_flip(image_arr, msk_arr):
    return np.flip(image_arr, 0), np.flip(msk_arr, 0)


def horizontal_flip(image_arr, msk_arr):
    return np.flip(image_arr, 1), np.flip(msk_arr, 1)


def hor_vert_flip(image_arr, msk_arr):
    ver_img = np.flip(image_arr, 0)
    ver_msk = np.flip(msk_arr, 0)

    ver_hor_img = np.flip(ver_img, 1)
    ver_hor_msk = np.flip(ver_msk, 1)

    return ver_hor_img, ver_hor_msk


def rotate_90(image_arr, msk_arr):
    rotated_img = transform.rotate(image_arr, 90)
    rotated_msk = transform.rotate(msk_arr, 90)

    rotated_img = rotated_img * 255.0
    rotated_img = rotated_img.astype(np.int8)

    return rotated_img, rotated_msk


def rotate_180(image_arr, msk_arr):
    rotated_img = transform.rotate(image_arr, 180)
    rotated_msk = transform.rotate(msk_arr, 180)

    rotated_img = rotated_img * 255.0
    rotated_img = rotated_img.astype(np.int8)

    return rotated_img, rotated_msk


def rotate_270(image_arr, msk_arr):
    rotated_img = transform.rotate(image_arr, 270)
    rotated_msk = transform.rotate(msk_arr, 270)

    rotated_img = rotated_img * 255.0
    rotated_img = rotated_img.astype(np.int8)

    return rotated_img, rotated_msk


def get_files(folder_path):
    # get all files
    images = glob.glob(
        os.path.join(folder_path, '*tif*')
    )

    return images


def get_img_arr(image_path):
    img_split = image_path.split('\\')[-1]
    img_name = img_split.replace('.tif', '')

    msk_path = image_path.replace('.tif', '_gt.bmp')
    # read image as an two dimensional array of pixels
    image_to_transform = imageio.imread(image_path)
    msk_to_transform = imageio.imread(msk_path)

    return image_to_transform, msk_to_transform, img_name


def flip(data_dir):
    """
    :param data_dir: directory locate data
    :return: None
    """
    # 0: represent vertical flip
    # 1: represent horizontal flip
    # 2: represent horizontal and vertical flip
    flip_trans = {
        '0': vertical_flip,
        '1': horizontal_flip,
        '2': hor_vert_flip
    }
    images = get_files(data_dir)
    for image in images:
        for key in flip_trans:
            img2trans, msk2trans, img_name = get_img_arr(image)
            img_transformed, msk_transformed = flip_trans[key](img2trans, msk2trans)

            print('flip ' + img_name)

            img_file_name = '{}/{}_{}.tif'.format(data_dir, img_name, key)
            msk_file_name = '{}/{}_{}_gt.bmp'.format(data_dir, img_name, key)

            imageio.imwrite(img_file_name, img_transformed)
            imageio.imwrite(msk_file_name, msk_transformed)


def rotate(data_dir):
    images = get_files(data_dir)
    for image in images:
        img2trans, msk2trans, img_name = get_img_arr(image)
        img_transformed, msk_transformed = rotate_90(img2trans, msk2trans)

        print('rotate ' + img_name)
        img_file_name = '{}/{}_{}.tif'.format(data_dir, img_name, 'rotate_90')
        msk_file_name = '{}/{}_{}_gt.bmp'.format(data_dir, img_name, 'rotate_90')

        imageio.imwrite(img_file_name, img_transformed)
        imageio.imwrite(msk_file_name, msk_transformed)


def apply_random_rotate(data_dir, num):
    images = get_files(data_dir)
    for image in images:
        for i in range(num):
            img2trans, msk2trans, img_name = get_img_arr(image)
            img_transformed, msk_transformed = random_rotation(img2trans, msk2trans)

            print('randomly rotate ' + img_name)
            img_file_name = '{}/{}_{}.tif'.format(data_dir, img_name, 'random_rotate'+str(i))
            msk_file_name = '{}/{}_{}_gt.bmp'.format(data_dir, img_name, 'random_rotate'+str(i))

            imageio.imwrite(img_file_name, img_transformed)
            imageio.imwrite(msk_file_name, msk_transformed)


def main():
    data_dir = './contest_data/'

    # do flip first
    flip(data_dir)

    # rotate 90 degrees
    rotate(data_dir)

    # do rotate randomly, generate 5 files per image
    apply_random_rotate(data_dir, 5)
    # images = get_files(data_dir)


if __name__ == '__main__':
    main()
