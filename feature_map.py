import os
import warnings
import numpy as np
import nibabel as nib
from keras.optimizers import Adam
from metrics import dice_coefficient_loss, dice_coefficient, IoU
from utils.json_process import load_json
from utils.crop_nifti import crop_img
from data_generator import DataGenerator
from keras import Model
import matplotlib.pyplot as plt


from model_fcn_5layers import fcn_3d as model
# from model_unet3d_5layers import unet_3d as model
# from model_attention_unet3d_5layers import attention_unet_3d as model
# from model_res_att import model_resatt as model
# from model_unetres import model_res as model

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


def pred_nifti_process(file_list, results, save_dir, index=1):
    name = save_dir.split('/')[-2]
    mid = len(results) // 2
    np.save(save_dir+name+'_'+str(index)+'.npy', results[mid])


def pred_generator(partition_holdout, file_list, dim, batch_size=1):
    index = 0
    while True:
        x = np.empty((batch_size, *dim))
        for i, item in enumerate(partition_holdout[index*batch_size: (index+1)*batch_size]):
            if 'reaffine' in item:
                img1 = item + '/' + 'norm_reaffine.nii'
            elif 'resize' in item:
                img1 = item + '/' + 'norm_resize.nii'
            elif 'resample' in item:
                img1 = item + '/' + 'norm_resample.nii'

            file_list.append(img1)
            img1 = nib.load(img1)
            cropped = crop_img(img1)
            img_array = np.array(cropped.dataobj)
            # print('img_array', img_array.shape)
            a = np.zeros(dim)
            # print('a shape', a.shape)
            a[:dim[0], :img_array.shape[0], :img_array.shape[1], :img_array.shape[2]] = img_array
            a = a.astype('float32') / a.max()
            x[i] = a

        index +=1
        yield x


def main():
    filename = './holdout_finetuning.json'
    # filename = './prediction_nifti/fsl_resize_test.json'
    partition_holdout = load_json(filename)
    model_finetuning = model(input_shape=(1, 160, 160, 192), n_base_filters=6, dropout_rate=0.3, n_labels=1, optimizer=Adam, initial_learning_rate=0.01, loss_function=dice_coefficient_loss, activation_name='sigmoid', gpu_num=2)
    weights_1 = './model_checkpoint_fcn_3d/model_200.hdf5'
    weights_2 = './model_checkpoint_unet_3d/model_200.hdf5'
    weights_3 = './model_checkpoint_attention_unet_3d/model_200.hdf5'
    weights_4 = './model_checkpoint_res_att/model_200.hdf5'
    weights_5 = './model_checkpoint_unetres_3d/model_200.hdf5'
    model_finetuning.load_weights(weights_1)
    # for i, layer in enumerate(model_finetuning.layers):
    #     print(i, layer.name, layer.output_shape)

    index = 1

    new_model = model_finetuning.layers[-2]
    new_model.compile(optimizer=Adam(0.01), loss=dice_coefficient_loss, metrics=['accuracy', IoU, dice_coefficient])    

    # for i, layer in enumerate(new_model.layers):
    #     print(i, layer.name, layer.output_shape)

    feature_map = Model(inputs=new_model.get_input_at(0), outputs=new_model.get_layer(index=index).output)
    feature_map.compile(optimizer=Adam(0.01), loss=dice_coefficient_loss, metrics=['accuracy', IoU, dice_coefficient])

    # evalparams = {'dim': (160, 160, 192),
    #           'batch_size': 2,
    #           'n_channels': 1,
    #           'n_labels': 1,
    #           'shuffle': True}
    #
    # evalgenerator = DataGenerator(partition_holdout, **evalparams)
    #
    # # evalgenerator = eval_generator(partition_holdout, **predparams)
    # print('evaluating...')
    # eva = feature_map.evaluate_generator(generator=evalgenerator, steps=np.floor(len(partition_holdout)/evalparams['batch_size']-1),verbose=1)
    # print('Testing dataset loss = {:.2f}%'.format(eva[0] * 100.0))
    # print('Testing dataset dice = {:.2f}%'.format(eva[3] * 100.0))

    file_list = []
    predparams = {'dim': (1, 160, 160, 192),
                 'batch_size': 2,
                 'file_list': file_list
                 }


    predgenerator = pred_generator(partition_holdout, **predparams)
    print('predicting...')
    results = feature_map.predict_generator(generator=predgenerator, steps=np.floor(len(partition_holdout)/predparams['batch_size']-1), verbose=1)
    print(file_list)
    print(results.shape)

    save_dir1 = './features_maps/feature_fcn/'
    save_dir2 = './features_maps/feature_unet3d/'
    save_dir3 = './features_maps/feature_att/'
    save_dir4 = './features_maps/feature_resatt/'
    save_dir5 = './features_maps/feature_unetres/'
    pred_nifti_process(file_list, results, save_dir1, index=index)


if __name__ == '__main__':
    main()







