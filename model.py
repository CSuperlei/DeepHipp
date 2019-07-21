from functools import partial
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, UpSampling3D, Activation, BatchNormalization, LeakyReLU, Add, SpatialDropout3D, Multiply
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from metrics import dice_coefficient_loss, IoU, dice_coefficient
K.set_image_dim_ordering('tf')
K.set_image_data_format('channels_first')

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate


def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3,3,3), activation=None, padding='same', strides=(1,1,1), instance_normalization=False):
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization." "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1)(layer)

    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)


def create_localization_module(input_layer, n_filters):
    convolution1 = create_convolution_block(input_layer, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1,1,1))
    return convolution2


def create_up_sampling_module(input_layer, n_filters, size=(2,2,2)):
    up_sample = UpSampling3D(size=size)(input_layer)
    convolution = create_convolution_block(up_sample, n_filters=n_filters)
    return convolution


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format='channels_first'):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
    return convolution2


def attention_block_3d(x, g , inter_channel=1):  #x, g, inter_channel
    # x, g = att_arg
    # print('x',x.shape)
    # print('g',g.shape)
    theta_x = Conv3D(inter_channel, (2,2,2), strides=(2,2,2))(x)
    # print('theta_x', theta_x.shape)
    phi_g = Conv3D(inter_channel, (1,1,1), strides=(1,1,1))(g)
    # print('phi_g', phi_g.shape)
    add = Add()([theta_x, phi_g])
    f = Activation('relu')(add)
    psi_f = Conv3D(1, (1,1,1), strides=(1,1,1))(f)
    sigm_psi_f = Activation('sigmoid')(psi_f)
    rate = UpSampling3D(size=(2,2,2))(sigm_psi_f)
    att_x = Multiply()([x, rate])
    # print('att_x', att_x.shape)
    return att_x


## 偏函数，用于代替原来函数中一部分参数
create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)


def model_desatt(input_shape=(1, 160, 160, 192), n_base_filters=6, dropout_rate=0.3, n_labels=1, optimizer=Adam, initial_learning_rate=0.01, loss_function=dice_coefficient_loss, activation_name='sigmoid', gpu_num=1):
    inputs = Input(input_shape)
    current_layer = inputs

    conv1 = create_convolution_block(current_layer, n_base_filters)
    context_output_layer_1 = create_context_module(conv1, n_base_filters, dropout_rate=dropout_rate)
    summation_layer_1 = Add()([conv1, context_output_layer_1])

    conv2 = create_convolution_block(summation_layer_1, 2*n_base_filters, strides=(2,2,2))
    context_output_layer_2 = create_context_module(conv2, 2*n_base_filters, dropout_rate=dropout_rate)
    summation_layer_2 = Add()([conv2, context_output_layer_2])

    conv3 = create_convolution_block(summation_layer_2, 4*n_base_filters, strides=(2,2,2))
    context_output_layer_3 = create_context_module(conv3, 4*n_base_filters, dropout_rate=dropout_rate)
    summation_layer_3 = Add()([conv3, context_output_layer_3])

    conv4 = create_convolution_block(summation_layer_3, 8*n_base_filters, strides=(2,2,2))
    context_output_layer_4 = create_context_module(conv4, 8*n_base_filters, dropout_rate=dropout_rate)
    summation_layer_4 = Add()([conv4, context_output_layer_4])

    '''
    最底层
    '''
    conv5 = create_convolution_block(summation_layer_4, 16*n_base_filters, strides=(2,2,2))
    context_output_layer_5 = create_context_module(conv5, 16*n_base_filters, dropout_rate=dropout_rate)
    summation_layer_5 = Add()([conv5, context_output_layer_5])

    '''
    上采用======================================================================
    '''

    att_1 = attention_block_3d(summation_layer_4, summation_layer_5, inter_channel=8*n_base_filters)

    up_sampling_1 = create_up_sampling_module(summation_layer_5, 8*n_base_filters)
    concat_1 = concatenate([att_1, up_sampling_1], axis=1)
    localization_output_1 = create_localization_module(concat_1, 8*n_base_filters)

    att_2 = attention_block_3d(summation_layer_3, localization_output_1, inter_channel=4*n_base_filters)

    up_sampling_2 = create_up_sampling_module(localization_output_1, 4*n_base_filters)
    concat_2 = concatenate([att_2, up_sampling_2], axis=1)
    localization_output_2 = create_localization_module(concat_2, 4*n_base_filters)

    segmentation_layer1 = create_convolution_block(localization_output_2, n_filters=n_labels, kernel=(1,1,1))

    att_3 = attention_block_3d(summation_layer_2, localization_output_2, inter_channel=2*n_base_filters)

    up_sampling_3 = create_up_sampling_module(localization_output_2, 2*n_base_filters)
    concat_3 = concatenate([att_3, up_sampling_3], axis=1)
    localization_output_3 = create_localization_module(concat_3, 2*n_base_filters)

    segmentation_layer2 = create_convolution_block(localization_output_3, n_filters=n_labels, kernel=(1,1,1))

    att_4 = attention_block_3d(summation_layer_1, localization_output_3, inter_channel=n_base_filters)

    up_sampling_4 = create_up_sampling_module(localization_output_3, n_base_filters)
    concat_4 = concatenate([att_4, up_sampling_4], axis=1)
    localization_output_4 = create_localization_module(concat_4, n_base_filters)

    segmentation_layer3 = create_convolution_block(localization_output_4, n_filters=n_labels, kernel=(1,1,1))

    output_up1 = UpSampling3D(size=(2,2,2))(segmentation_layer1)
    output_concat_1 = Add()([output_up1, segmentation_layer2])

    output_up2 = UpSampling3D(size=(2,2,2))(output_concat_1)
    output_concat_2 = Add()([output_up2, segmentation_layer3])

    activation_block = Activation(activation_name)(output_concat_2)

    model = Model(inputs=inputs, outputs=activation_block)
    if gpu_num>1:
        model = multi_gpu_model(model, gpus=gpu_num)
    model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function, metrics=['accuracy', IoU, dice_coefficient])
    return model


if __name__ == '__main__':
    model = model_desatt(input_shape=(4, 160, 160, 192), n_base_filters=6, dropout_rate=0.3, n_labels=3, optimizer=Adam, initial_learning_rate=0.01, loss_function=dice_coefficient_loss, activation_name="sigmoid")
    model.summary()
    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.output_shape)

























