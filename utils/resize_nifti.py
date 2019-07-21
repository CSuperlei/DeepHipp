import numpy as np
import nibabel as nib
from nilearn.image import reorder_img, new_img_like
import SimpleITK as sitk
from json_process import load_json
from affine_register import affine_register


def data_to_sitk_image(data, spacing=(1., 1., 1.)):
    if len(data.shape) == 3:
        data = np.rot90(data, 1, axes=(0, 2))  ## 逆时针旋转, 把0轴和2轴交换

    image = sitk.GetImageFromArray(data)
    image.SetSpacing(np.asarray(spacing, dtype=np.float))
    return image


def calculate_origin_offset(new_spacing, old_spacing):
    return np.subtract(new_spacing, old_spacing)/2


def sitk_new_blank_image(size, spacing, direction, origin, default_value=0.):
    image = sitk.GetImageFromArray(np.ones(size, dtype=np.float).T * default_value)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    image.SetOrigin(origin)
    return image


def sitk_resample_to_image(image, reference_image, interpolator=sitk.sitkLinear, default_value=0., transform=None, output_pixel_type=None):
    if transform is None:
        transform = sitk.Transform()
        transform.SetIdentity()
    if output_pixel_type is None:
        output_pixel_type = image.GetPixelID()

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetInterpolator(interpolator)
    resample_filter.SetTransform(transform)
    resample_filter.SetOutputPixelType(output_pixel_type)
    resample_filter.SetDefaultPixelValue(default_value)
    resample_filter.SetReferenceImage(reference_image)
    return resample_filter.Execute(image)


def sitk_resample_to_spacing(image, new_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear, default_value=0.):
    zoom_factor = np.divide(image.GetSpacing(), new_spacing)
    new_size = np.asarray(np.ceil(np.round(np.multiply(zoom_factor, image.GetSize()), decimals=5)), dtype=np.int16)
    offset = calculate_origin_offset(new_spacing, image.GetSpacing())
    reference_image = sitk_new_blank_image(size=new_size, spacing=new_spacing, direction=image.GetDirection(), origin=image.GetOrigin()+offset, default_value=default_value)
    return sitk_resample_to_image(image, reference_image, interpolator=interpolator, default_value=default_value)


def sitk_image_to_data(image):
    data = sitk.GetArrayFromImage(image)
    if len(data.shape) == 3:
        data = np.rot90(data, -1, axes=(0,2))
    return data


def resample_to_spacing(data, spacing, target_spacing, interpolation='linear', default_value=0):
    '''

    :param data:
    :param spacing:
    :param new_spacing:
    :param interpolation:
    :param default_value:
    :return:
    '''
    image = data_to_sitk_image(data, spacing=spacing)
    if interpolation is 'linear':
        interpolator = sitk.sitkLinear
    elif interpolation is 'nearest':
        interpolator = sitk.sitkNearestNeighbor
    else:
        raise ValueError("'interpolation' must be either 'linear' or 'nearest'. '{}' is not recognized".format(
            interpolation))
    resampled_image = sitk_resample_to_spacing(image, new_spacing=target_spacing, interpolator=interpolator, default_value=default_value)
    return sitk_image_to_data(resampled_image)


def resize(image, new_shape, interpolation='linear'):
    image = reorder_img(image, resample=interpolation)
    # print('reorder_img', image)
    # print(image.shape)
    ## 两次相除
    zoom_level = np.divide(new_shape, image.shape)
    new_spacing = np.divide(image.header.get_zooms(), zoom_level)
    # print('new_spacing', new_spacing)
    new_data = resample_to_spacing(image.get_data(), image.header.get_zooms(), new_spacing, interpolation=interpolation)
    new_affine = np.copy(image.affine)
    np.fill_diagonal(new_affine, new_spacing.tolist()+[1])
    new_affine[:3, 3] += calculate_origin_offset(new_spacing, image.header.get_zooms())
    return new_img_like(image, new_data, affine=new_affine)


def resize_image(fileitems, resize_shape=(160, 160, 192)):
    for item in fileitems:
        img1 = item + '/normalize/' + 'norm.nii'
        img2 = item + '/normalize/' + 'nu.nii'
        img3 = item + '/normalize/' + 'lh.hippoSfLabels-T1.v10.FSvoxelSpace.nii'
        img4 = item + '/normalize/' + 'rh.hippoSfLabels-T1.v10.FSvoxelSpace.nii'

        img1 = nib.load(img1)
        img2 = nib.load(img2)
        img3 = nib.load(img3)
        img4 = nib.load(img4)

        norm_resize = resize(img1, resize_shape)
        nu_resize = resize(img2, resize_shape)
        lhip_resize = resize(img3, resize_shape)
        rhip_resize = resize(img4, resize_shape)

        norm_resize = affine_register(norm_resize)
        nu_resize = affine_register(nu_resize)
        lhip_resize = affine_register(lhip_resize)
        rhip_resize = affine_register(rhip_resize)

        norm_data = nib.AnalyzeImage(norm_resize.get_data(), norm_resize.affine)
        nu_data = nib.AnalyzeImage(nu_resize.get_data(), nu_resize.affine)
        lhip_data = nib.AnalyzeImage(lhip_resize.get_data(), lhip_resize.affine)
        rhip_data = nib.AnalyzeImage(rhip_resize.get_data(), rhip_resize.affine)

        nib.save(norm_data, item+'/resize/'+'norm_resize.nii')
        nib.save(nu_data, item+'/resize/'+'nu_resize.nii')
        nib.save(lhip_data, item+'/resize/'+'lhipp_resize.nii')
        nib.save(rhip_data, item+'/resize/'+'rhipp_resize.nii')
        # break


if __name__ == '__main__':
    # file_origin = 'E:\\PythonProjects\\Medical Image 2019\\Isensee2017\\beta_finetuning_new\\file_origin.json'
    file_origin = '../file_origin.json'
    fileitems = load_json(file_origin)
    resize_image(fileitems=fileitems)
