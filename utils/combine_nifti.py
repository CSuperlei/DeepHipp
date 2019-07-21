import numpy as np
import nibabel as nib
from json_process import load_json
from affine_register import affine_register


def combine_nifti(fileitems):
    for temp in fileitems:
        tag = temp.split('/')[-1]
        lhip_name = temp + '/lhipp_'+tag+'.nii'
        rhip_name = temp + '/rhipp_'+tag+'.nii'
        lhip_img = nib.load(lhip_name)
        rhip_img = nib.load(rhip_name)
        lhip_data = np.asarray(lhip_img.get_data())
        rhip_data = np.asarray(rhip_img.get_data())
        hip_data = lhip_data + rhip_data
        if (lhip_img.affine == rhip_img.affine).all():
            lhip_img = affine_register(lhip_img)
            hip_data = nib.AnalyzeImage(hip_data, lhip_img.affine)
            nib.save(hip_data, temp+'/hipp_combine_'+tag+'.nii')
        else:
            print('affine is not equal')
        # break


if __name__ == '__main__':
    filename1 = '../holdout_finetuning.json'
    filename2 = '../train_finetuning_list.json'
    filename3 = '../test_finetuning_list.json'
    fileitem1 = load_json(filename1)
    fileitem2 = load_json(filename2)
    fileitem3 = load_json(filename3)
    fileitems = fileitem1 + fileitem2 + fileitem3
    combine_nifti(fileitems=fileitems)



