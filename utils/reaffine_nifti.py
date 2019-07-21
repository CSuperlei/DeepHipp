import numpy as np
import nibabel as nib
from nilearn.image.image import new_img_like
from json_process import load_json
from affine_register import affine_register


def curt_nifti_image(fileitems):
    for item in fileitems:
        img1 = item + '/normalize/' + 'norm.nii'
        img2 = item + '/normalize/' + 'nu.nii'
        img3 = item + '/normalize/' + 'lh.hippoSfLabels-T1.v10.FSvoxelSpace.nii'
        img4 = item + '/normalize/' + 'rh.hippoSfLabels-T1.v10.FSvoxelSpace.nii'

        img_norm = nib.load(img1)
        data_norm = img_norm.get_data()
        nozero = np.nonzero(data_norm)
        center_x = (nozero[0].max() + nozero[0].min()) / 2
        center_y = (nozero[1].max() + nozero[1].min()) / 2
        center_z = (nozero[2].max() + nozero[2].min()) / 2
        box = np.zeros((3, 2), dtype=int)
        box[0, 0] = int(center_x - 80)
        box[0, 1] = int(center_x + 80)
        box[1, 0] = int(center_y - 80)
        box[1, 1] = int(center_y + 80)
        box[2, 0] = int(center_z - 96)
        box[2, 1] = int(center_z + 96)
        norm_result = data_norm[box[0,0]:box[0,1], box[1,0]:box[1,1], box[2,0]:box[2,1]]
        img_norm = affine_register(img_norm)
        norm_result = nib.AnalyzeImage(norm_result, img_norm.affine)
        nib.save(norm_result, item+'/reaffine/'+'norm_reaffine.nii')


        img_nu = nib.load(img2)
        data_nu = img_nu.get_data()
        nu_result = data_nu[box[0,0]:box[0,1], box[1,0]:box[1,1], box[2,0]:box[2,1]]
        img_nu = affine_register(img_nu)
        nu_result = nib.AnalyzeImage(nu_result, img_nu.affine)
        nib.save(nu_result, item+'/reaffine/'+'nu_reaffine.nii')

        img_lhip = nib.load(img3)
        data_lhip = img_lhip.get_data()
        lhip_result = data_lhip[box[0,0]:box[0,1], box[1,0]:box[1,1], box[2,0]:box[2,1]]
        img_lhip = affine_register(img_lhip)
        lhip_result = nib.AnalyzeImage(lhip_result, img_lhip.affine)
        nib.save(lhip_result, item+'/reaffine/'+'lhipp_reaffine.nii')

        img_rhip = nib.load(img4)
        data_rhip = img_rhip.get_data()
        rhip_result = data_rhip[box[0,0]:box[0,1], box[1,0]:box[1,1], box[2,0]:box[2,1]]
        img_rhip = affine_register(img_rhip)
        rhip_result = nib.AnalyzeImage(rhip_result, img_rhip.affine)
        nib.save(rhip_result, item+'/reaffine/'+'rhipp_reaffine.nii')
        # break


if __name__ == '__main__':
    # file_origin = 'E:\\PythonProjects\\Medical Image 2019\\Isensee2017\\beta_finetuning_new\\file_origin.json'
    file_origin = '../file_origin.json'
    fileitems = load_json(file_origin)
    curt_nifti_image(fileitems=fileitems)
