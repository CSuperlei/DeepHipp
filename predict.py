import os
import warnings
from keras.optimizers import Adam
from data_generator import DataGenerator
from model import model_desatt, dice_coefficient_loss
from utils.json_process import load_json
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"


def main():
    filename = './holdout_finetuning.json'
    partition_holdout = load_json(filename)

    params = {'dim': (160, 160, 192),
              'batch_size': 4,
              'n_channels': 1,
              'n_labels': 1,
              'shuffle': False}

    prediction_generator = DataGenerator(partition_holdout, **params)

    model_finetuning = model(input_shape=(1, 160, 160, 192), n_base_filters=6, dropout_rate=0.3, n_labels=1, optimizer=Adam, initial_learning_rate=0.01, loss_function=dice_coefficient_loss, activation_name='sigmoid', gpu_num=4)


    # '/home/cailei/isensee2017/beta_channelfirst_finetuning/model_checkpoint_200_300'
    model_finetuning.load_weights('./model_checkpoint_newdata_combinee/model_200.hdf5')

    print('evaluating...')
    eva = model_finetuning.evaluate_generator(prediction_generator, verbose=1)
    print('Testing dataset loss = {:.2f}%'.format(eva[0] * 100.0))
    print('Testing dataset accuracy = {:.2f}%'.format(eva[1] * 100.0))
    print('Testing dataset IoU = {:.2f}%'.format(eva[2] * 100.0))
    print('Testing dataset dice = {:.2f}%'.format(eva[3] * 100.0))

    # results = model_finetuning.predict_generator(generator=prediction_generator, verbose=1)
    # print(results.shape)
    #
    # np.save('./predict_hipp_combine_newdata_3dunet.npy', results)


if __name__ == '__main__':
    main()


