import os
import time
import warnings
import keras
from keras.optimizers import Adam
from data_generator import DataGenerator
from utils.json_process import load_json
from metrics import dice_coefficient_loss
from model import model_desatt as model
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,1,2,3,4"


# auto lr
def lr_step_decay(epoch):
    initial_lr = 0.01
    drop = 10
    epochs_drop = 40
    lr = initial_lr / pow(drop, int((epoch - 1) / epochs_drop))
    return lr


def main():
    filename_train = './train_finetuning_list.json'
    training_generator = load_json(filename_train)
    filename_test = './test_finetuning_list.json'
    test_generator = load_json(filename_test)

    params = {'dim': (160,160,192),
              'batch_size':2,
              'n_channels':1,
              'n_labels' :1,
              'shuffle': True}


    training_generator = DataGenerator(training_generator, **params)
    validation_generator = DataGenerator(test_generator, **params)

    #cb_1 = keras.callbacks.EarlyStopping(min_delta=0, patience=30, verbose=0, mode='auto')
    cb_2 = keras.callbacks.ModelCheckpoint(filepath='./model_checkpoint_unet_3d/model_{epoch:03d}.hdf5', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    model_name = 'tensorboard_unet_3d_{}'.format(int(time.time()))
    cb_3 = keras.callbacks.TensorBoard(log_dir='./tensorboard_logs/{}'.format(model_name))
    cb_4 = keras.callbacks.LearningRateScheduler(lr_step_decay, verbose=1)
    cb_5 = keras.callbacks.ReduceLROnPlateau(factor=0.5, verbose=1, patience=50)

    ## 做一下数据轴变换
    model_finetuning = model(input_shape=(1, 160, 160, 192), n_base_filters=6, dropout_rate=0.3, n_labels=1, optimizer=Adam, initial_learning_rate=0.01, loss_function=dice_coefficient_loss, activation_name='sigmoid', gpu_num=2)

    results = model_finetuning.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  epochs=200,
                                  nb_worker=4,
                                  callbacks=[cb_2, cb_3, cb_4, cb_5])


if __name__ == '__main__':
    main()

