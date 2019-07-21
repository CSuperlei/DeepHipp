import numpy as np
import nibabel as nib
from nilearn.image import new_img_like, resample_to_img
import random
import itertools
from json_process import load_json
from affine_register import affine_register


def random_scale_factor(n_dim=3, std=0.25, mean=1):
    return np.random.normal(mean, std, n_dim)   ## 产生3个均值为1, 标准差为0.25的数


def random_boolean():
    return np.random.choice([True, False])


def random_flip_dimensions(n_dimensions):
    axis = list()
    for dim in range(n_dimensions):
        if random_boolean():
            axis.append(dim)

    return axis  ##[0,1,2]


def get_image(data, affine, nib_class=nib.Nifti1Image):
    return nib_class(dataobj=data, affine=affine)


def flip_image(image, axis):
    '''

    :param image:
    :param axis: [0, 1,2]
    :return:
    '''
    try:
        new_data = np.copy(image.get_data())
        for axis_index in axis:
            new_data = np.flip(new_data, axis=axis_index)
    except TypeError:
        new_data = np.flip(image.get_data(), axis=axis)

    return new_img_like(image, new_data)


def scale_image(image, scale_factor):
    scale_factor = np.asarray(scale_factor)
    new_affine = np.copy(image.affine)
    ## 对像素空间进行压缩，数据大小不变
    new_affine[:3, :3] = image.affine[:3, :3]*scale_factor
    new_affine[:, 3][:3] = image.affine[:, 3][:3] + (image.shape*np.diag(image.affine)[:3]*(1-scale_factor)) / 2

    return new_img_like(image, data=image.get_data(), affine=new_affine)


def distort_image(image, flip_axis=None, scale_factor=None):
    if flip_axis:
        image = flip_image(image, flip_axis)
    if scale_factor is not None:
        image = scale_image(image, scale_factor)

    return image


def augment_data(data_1, data_2, truth_1, truth_2, affine, scale_deviation=None, flip=True):
    n_dim = len(truth_1.shape)  ## 三维n_dim=3
    if scale_deviation:
        scale_factor = random_scale_factor(n_dim, std=scale_deviation)
    else:
        print('scale_factor None')
        scale_factor = None
    if flip:
        flip_axis = random_flip_dimensions(n_dim)
    else:
        print('flip_axis None')
        flip_axis = None

    image_1 = get_image(data_1, affine)
    distort_img_1 = distort_image(image_1, flip_axis=flip_axis, scale_factor=scale_factor)
    # data_1 = resample_to_img(distort_img_1, image_1, interpolation='continuous').get_data()
    data_1 = resample_to_img(distort_img_1, image_1, interpolation='continuous')

    image_2 = get_image(data_2, affine)
    distort_img_2 = distort_image(image_2, flip_axis=flip_axis, scale_factor=scale_factor)
    # data_2 = resample_to_img(distort_img_2, image_2, interpolation='continuous').get_data()
    data_2 = resample_to_img(distort_img_2, image_2, interpolation='continuous')

    truth_image_1 = get_image(truth_1, affine)
    distort_truth_1 = distort_image(truth_image_1, flip_axis=flip_axis, scale_factor=scale_factor)
    # truth_data_1 = resample_to_img(distort_truth_1, truth_image_1, interpolation='nearest').get_data()
    truth_data_1 = resample_to_img(distort_truth_1, truth_image_1, interpolation='nearest')

    truth_image_2 = get_image(truth_2, affine)
    distort_truth_2 = distort_image(truth_image_2, flip_axis=flip_axis, scale_factor=scale_factor)
    # truth_data_2 = resample_to_img(distort_truth_2, truth_image_2, interpolation='nearest').get_data()
    truth_data_2 = resample_to_img(distort_truth_2, truth_image_2, interpolation='nearest')

    return data_1, data_2, truth_data_1, truth_data_2

'''
===============================================================================
permute x y
'''

def generate_permutation_keys():
    return set(itertools.product(itertools.combinations_with_replacement(range(2), 2), range(2), range(2), range(2),range(2)))


def random_permutation_key():
    return random.choice(list(generate_permutation_keys()))


def permute_data(data, key):
    data = np.copy(data)
    (rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose = key
    if rotate_y != 0:
        ## np.rot90(数组, 旋转次数, 交换的轴)  axes=(1,3) 说明1,3轴交换，也代表旋转了
        ## 例如数组是(1,160,160,192) axes=(1,3) 旋转完就变成(1,192,160,160)
        data = np.rot90(data, rotate_y, axes=(1,3))
    if rotate_z != 0:
        data = np.rot90(data, rotate_z, axes=(2,3))
    if flip_x:
        ## 将最后一个数与第一个数交换
        data = data[:, ::-1]
    if flip_y:
        data = data[:, :, ::-1]
    if flip_z:
        ## 四维数据才用到，三维用不到
        data = data[:, :, :, ::-1]
    if transpose:
        ## 三维直接data = data.T即可
        for i in range(data.shape[0]):
            data[i] = data[i].T
    return data


def random_permutation_x_y(x_data, y_data):
    key = random_permutation_key()
    return permute_data(x_data, key), permute_data(y_data, key)


def augment_image(fileitems):
    for item in fileitems:
        img1 = item + '/resize/' + 'norm_resize.nii'
        img2 = item + '/resize/' + 'nu_resize.nii'
        img3 = item + '/resize/' + 'lhipp_resize.nii'
        img4 = item + '/resize/' + 'rhipp_resize.nii'

        img1 = nib.load(img1)
        data_1 = img1.get_data()
        img2 = nib.load(img2)
        data_2 = img2.get_data()
        img3 = nib.load(img3)
        truth_1 = img3.get_data()
        img4 = nib.load(img4)
        truth_2 = img4.get_data()

        data_1, data_2, truth_1, truth_2 = augment_data(data_1, data_2, truth_1, truth_2, img1.affine, scale_deviation=0.25, flip=True)

        data_1 = affine_register(data_1)
        data_2 = affine_register(data_2)
        truth_1 = affine_register(truth_1)
        truth_2 = affine_register(truth_2)

        # data_1 = nib.AnalyzeImage(data_1, img1.affine)
        # data_2 = nib.AnalyzeImage(data_2, img2.affine)
        # truth_1 = nib.AnalyzeImage(truth_1, img3.affine)
        # truth_2 = nib.AnalyzeImage(truth_2, img4.affine)

        nii_data_1 = nib.AnalyzeImage(data_1.get_data(), data_1.affine)
        nii_data_2 = nib.AnalyzeImage(data_2.get_data(), data_2.affine)
        nii_truth_1 = nib.AnalyzeImage(truth_1.get_data(), truth_1.affine)
        nii_truth_2 = nib.AnalyzeImage(truth_2.get_data(), truth_2.affine)

        nib.save(nii_data_1, item+'/resample/'+'norm_resample.nii')
        nib.save(nii_data_2, item+'/resample/'+'nu_resample.nii')
        nib.save(nii_truth_1, item+'/resample/'+'lhipp_resample.nii')
        nib.save(nii_truth_2, item+'/resample/'+'rhipp_resample.nii')

        # break



if __name__ == '__main__':
    # file_origin = 'E:\\PythonProjects\\Medical Image 2019\\Isensee2017\\beta_finetuning_new\\file_origin.json'
    file_origin = '../file_origin.json'
    fileitems = load_json(file_origin)
    augment_image(fileitems)