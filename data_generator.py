import keras
import nibabel as nib
import numpy as np
from utils.crop_nifti import crop_img
from utils.affine_register import affine_register


class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, batch_size=2, dim=(160,160,192), n_channels=3, n_labels=1, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_labels = n_labels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        ## 每训练一轮一共有都少个batch size
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        ## 构造一个[0,....,sampls]得列表
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size]

        ## 根据索引找到每条索引对应得文件名
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        ##根据文件名加载数据
        X, y1 = self.__data_generation(list_IDs_temp)

        return X, y1


    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, self.n_channels, *self.dim))
        print('X.shape', X.shape)
        y1 = np.empty((self.batch_size, self.n_labels, *self.dim))
        print('y1.shape', y1.shape)
        for i, item in enumerate(list_IDs_temp):
            if 'reaffine' in item:
                img1 = item + '/' + 'norm_reaffine.nii'
                img2 = item + '/' + 'hipp_combine_reaffine.nii'
                ii = nib.load(img2)
                print('ii', ii.get_data().shape)
                print('img1', img1)
                print('img2_seg', img2)
            elif 'resize' in item:
                img1 = item + '/' + 'norm_resize.nii'
                img2 = item + '/' + 'hipp_combine_resize.nii'
                ii = nib.load(img2)
                print('ii', ii.get_data().shape)
                print('img1', img1)
                print('img2_seg', img2)
            elif 'resample' in item:
                img1 = item + '/' + 'norm_resample.nii'
                img2 = item + '/' + 'hipp_combine_resample.nii'
                ii = nib.load(img2)
                print('ii', ii.get_data().shape)
                print('img1', img1)
                print('img2_seg', img2)

            img1 = nib.load(img1)
            img2 = nib.load(img2)

            if ((img1.affine != img2.affine).all()):
                print('affine is not equal')

            img1 = affine_register(img1)
            img2 = affine_register(img2)
            # print(img1.affine)
            # print(img2.affine)
            newimage = nib.concat_images([img1, img2])
            cropped = crop_img(newimage)
            print('cropped', cropped.shape)
            img_array = np.array(cropped.dataobj)
            print('img_array', img_array.shape)
            ## 将图片的channel,与第一个维度互换
            ## 将2,3通道也要互换一下 原始(3, 160, 160, 192) -> (3, 160, 192, 160)
            z = np.rollaxis(img_array, 3, 0)
            # z = np.swapaxes(z, 2, 3)
            print('z shape', z.shape)

            padded_image = np.zeros((2, *self.dim))
            padded_image[:z.shape[0], :z.shape[1], :z.shape[2], :z.shape[3]] = z

            ## 将padded_image沿第0轴分为2组
            a, seg_mask_hipp = np.split(padded_image, 2, axis=0)

            ## 归一化

            a = a.astype('float32') / a.max()
            # a = (a.astype('float32') - np.mean(a.astype('float32'))) / np.std(a.astype('float32'))

            seg_mask = np.zeros((1, *self.dim))
            seg_mask[seg_mask_hipp.astype(int)>0] = 1

            X[i, ] = a
            y1[i, ] = seg_mask

        return X, y1





