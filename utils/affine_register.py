import numpy as np
from nilearn.image import new_img_like


def affine_register(img_affine):
    new_affine = np.copy(np.rint(img_affine.affine).astype('int'))
    img_new = new_img_like(img_affine, img_affine.get_data(), affine=new_affine)
    return img_new
