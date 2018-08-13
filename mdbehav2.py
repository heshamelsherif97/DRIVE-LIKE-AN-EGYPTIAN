import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from utils import INPUT_SHAPE, batch_generator
import argparse
import os
import pickle
from keras.utils.vis_utils import plot_model
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

np.random.seed(0)


def load_data(args):
    """
    Load training data and split it into training and validation set
    """

    data_df = pd.read_csv(os.path.join(args.data_dir, 'driving_log1.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed', 'rpm', 'sens1', 'sens2', 'sens3', 'sens4'
    , 'sens5', 'sens6', 'sens7', 'sens8', 'sens9', 'sens10', 'sens11', 'sens12', 'sens13', 'sens14', 'sens15', 'sens16', 'sens17', 'sens18', 'sens19', 'sens20', 'sens21'
    , 'sens22', 'sens23', 'sens24', 'sens25', 'sens26', 'sens27', 'sens28', 'sens29', 'sens30'])


    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values

    print(X[0])
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid


def build_model(args):
    """
    Modified NVIDIA model
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    checkpoint = ModelCheckpoint('modelBehavioralObs-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate), metrics=['accuracy'])

    return model.fit_generator(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
                        args.samples_per_epoch,
                        validation_steps=len(X_valid),
                        validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                        verbose=1,
                        callbacks=[checkpoint],
                        max_queue_size=1,
                        epochs=10)


def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Imititation Training')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=20000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    data = load_data(args)
    model = build_model(args)
    history = train_model(model, args, *data)
    filename = open("pltdataforbehavioralOvbs" + ".pickle","wb")
    pickle.dump(history.history, filename)
    filename.close()


if __name__ == '__main__':
    main()
