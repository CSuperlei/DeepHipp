import os
import nibabel as nib
from utils.json_process import load_json, dump_json

def file_mkdir(file_dir):
    brain_file_names = []
    for root, dirs, files in os.walk(file_dir, topdown=True):
        for name in dirs:
            directory_name = os.path.join(root, name)
            if 'normalize' not in directory_name:
                print(directory_name)
                brain_file_names.append(directory_name)

    for temp in brain_file_names:
        new_temp_reaffine = temp+'/reaffine'
        new_temp_resize = temp+'/resize'
        new_temp_resample = temp+'/resample'
        os.makedirs(new_temp_reaffine)
        os.makedirs(new_temp_resize)
        os.makedirs(new_temp_resample)


def file_top_travel(file_dir):
    file_names = []
    for root, dirs, files in os.walk(file_dir, topdown=True):
        for name in dirs:
            directory_name = os.path.join(root, name)
            if 'normalize' not in directory_name and 'reaffine' not in directory_name and 'resize' not in directory_name and 'resample' not in directory_name :
                file_names.append(directory_name)

    print(file_names)
    print(len(file_names))
    dump_json('./file_origin.json', file_names)
    return file_names


def file_travel(file_dir):
    brain_file_names = []
    for root, dirs, files in os.walk(file_dir, topdown=True):
        for name in dirs:
            directory_name = os.path.join(root, name)
            if 'reaffine' in directory_name:
                brain_file_names.append(directory_name)
            elif 'resize' in directory_name:
                brain_file_names.append(directory_name)
            elif 'resample' in directory_name:
                brain_file_names.append(directory_name)

    # print(brain_file_names)
    print(len(brain_file_names))
    return brain_file_names


def data_set_split(brain_filenames, holdout_percentage=0.15, train_percentage=0.7):
    partition = {}
    # holdout_percentage = 0.15
    partition['holdout'] = brain_filenames[0: int(len(brain_filenames)*holdout_percentage)]
    train_list = brain_filenames[int(len(brain_filenames)*holdout_percentage) : len(brain_filenames)]

    # train_percentage = 0.7
    partition['train'] = train_list[0:int(len(train_list)*train_percentage)]
    partition['test'] = train_list[int(len(train_list)*train_percentage) : len(train_list)]

    ## 存成json
    dump_json('./train_finetuning_list.json', partition['train'])
    dump_json('./test_finetuning_list.json', partition['test'])
    dump_json('./holdout_finetuning.json', partition['holdout'])

    print(partition)
    return partition


def remove_redundant_data(fileitems):
    for item in fileitems:
        img1 = item + '/' + 'lhipp_resize_affine.nii'
        img2 = item + '/' + 'lhipp_resize.nii'
        img3 = item + '/' + 'norm_resize_affine.nii'
        img4 = item + '/' + 'norm_resize.nii'
        img5 = item + '/' + 'nu_resize_affine.nii'
        img6 = item + '/' + 'nu_resize.nii'
        img7 = item + '/' + 'rhipp_resize_affine.nii'
        img8 = item + '/' + 'rhipp_resize.nii'
        os.remove(img1)
        os.remove(img2)
        os.remove(img3)
        os.remove(img4)
        os.remove(img5)
        os.remove(img6)
        os.remove(img7)
        os.remove(img8)


if __name__ == '__main__':
    # file_dir = 'e:\\lrhip\\'
    file_dir = '/home/cailei/lrhip'
    file_mkdir(file_dir)
    brain_filenames = file_travel(file_dir)
    data_set_split(brain_filenames)
    file_top_travel(file_dir)
