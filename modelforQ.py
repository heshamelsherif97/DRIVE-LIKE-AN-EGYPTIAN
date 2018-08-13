import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from utilsQ import INPUT_SHAPE, batch_generator, load_image, preprocess
import argparse
import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
import pickle


from keras.models import load_model
# from keras.utils import plot_model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

np.random.seed(0)


def getRightorLeft(x):
    if (x > 0 ):
        return "right"
    elif (x < 0):
        return "left"
    elif (x == 0):
        return "nodir"
    else:
        return ""


def getUporDown(x):
    if (x > 0):
        return "acc"
    else:
        return ""

def getAction(x):
    if (x == "accnodir"):
        return 0
    elif (x == "accright"):
        return 1
    elif (x == "accleft"):
        return 2
    else:
        return 3



def load_data(args, model):
    """
    Load training data and split it into training and validation set
    """

    actions = []

    data_df = pd.read_csv(os.path.join(args.data_dir, 'driving_log.csv'), names=['center', 'center2', 'center3', 'center4', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed', 'rpm', 'sens1', 'sens2', 'sens3', 'sens4'
    , 'sens5', 'sens6', 'sens7', 'sens8', 'sens9', 'sens10', 'sens11', 'sens12', 'sens13', 'sens14', 'sens15', 'sens16', 'sens17', 'sens18', 'sens19', 'sens20', 'sens21'
    , 'sens22', 'sens23', 'sens24', 'sens25', 'sens26', 'sens27', 'sens28', 'sens29', 'sens30'])

    X = data_df['center1'].values
    X2 = data_df['center1'].values
    X3 = data_df['center1'].values
    X4 = data_df['center1'].values
    y = data_df['steering'].values
    w = data_df['throttle'].values
    q = data_df['reverse'].values
    u = data_df['rpm'].values
    s = data_df['speed'].values
    se1 = data_df['sens1'].values
    se2 = data_df['sens2'].values
    se3 = data_df['sens3'].values
    se4 = data_df['sens4'].values
    se5 = data_df['sens5'].values
    se6 = data_df['sens6'].values
    se7 = data_df['sens7'].values
    se8 = data_df['sens8'].values
    se9 = data_df['sens9'].values
    se10 = data_df['sens10'].values
    se11 = data_df['sens11'].values
    se12 = data_df['sens12'].values
    se13 = data_df['sens13'].values
    se14 = data_df['sens14'].values
    se15 = data_df['sens15'].values
    se16 = data_df['sens16'].values
    se17 = data_df['sens17'].values
    se18 = data_df['sens18'].values
    se19 = data_df['sens19'].values
    se20 = data_df['sens20'].values
    se21 = data_df['sens21'].values
    se22 = data_df['sens22'].values
    se23 = data_df['sens23'].values
    se24 = data_df['sens24'].values
    se25 = data_df['sens25'].values
    se26 = data_df['sens26'].values
    se27 = data_df['sens27'].values
    se28 = data_df['sens28'].values
    se29 = data_df['sens29'].values
    se30 = data_df['sens30'].values

    action = 0
    actionlist = []
    actions = []
    imgs = np.empty([int(len(X)), 84, 84, 4])

    states = []

    for i in range(len(y)):
        state = []
        action = getAction(getUporDown(w[i])+getRightorLeft(y[i]))
        actionlist = []
        for z in range(0,4):
            if action == z :
                actionlist.append(float(1))
            else :
                actionlist.append(float(0))
        state.append(float(y[i]))
        state.append(float(w[i]))
        state.append(float(q[i]))
        state.append(float(u[i]))
        state.append(float(s[i]))
        state.append(float(se1[i]))
        state.append(float(se2[i]))
        state.append(float(se3[i]))
        state.append(float(se4[i]))
        state.append(float(se5[i]))
        state.append(float(se6[i]))
        state.append(float(se7[i]))
        state.append(float(se8[i]))
        state.append(float(se9[i]))
        state.append(float(se10[i]))
        state.append(float(se11[i]))
        state.append(float(se12[i]))
        state.append(float(se13[i]))
        state.append(float(se14[i]))
        state.append(float(se15[i]))
        state.append(float(se16[i]))
        state.append(float(se17[i]))
        state.append(float(se18[i]))
        state.append(float(se19[i]))
        state.append(float(se20[i]))
        state.append(float(se21[i]))
        state.append(float(se22[i]))
        state.append(float(se23[i]))
        state.append(float(se24[i]))
        state.append(float(se25[i]))
        state.append(float(se26[i]))
        state.append(float(se27[i]))
        state.append(float(se28[i]))
        state.append(float(se29[i]))
        state.append(float(se30[i]))
        states.append(state)
        actions.append(actionlist)
        image = Image.open(X[i])
        image = np.asarray(image)       # from PIL image to numpy array
        image = preprocess(image) # apply the preprocessing
        image = image.reshape(1, 84, 84)       #
        image2 = Image.open(X2[i])
        image2 = np.asarray(image2)       # from PIL image to numpy array
        image2 = preprocess(image2) # apply the preprocessing
        image2 = image2.reshape(1, 84, 84)
        image3 = Image.open(X3[i])
        image3 = np.asarray(image3)       # from PIL image to numpy array
        image3 = preprocess(image3) # apply the preprocessing
        image3 = image3.reshape(1, 84, 84)
        image4 = Image.open(X4[i])
        image4 = np.asarray(image4)       # from PIL image to numpy array
        image4 = preprocess(image4) # apply the preprocessing
        image4 = image4.reshape(1, 84, 84)
        input_img = np.stack((image, image2, image3, image4), axis = 3)
        imgs[i] = input_img


    states= np.array(states)
    actions = np.array(actions)
    print("training")
    history = model.fit([imgs, states], actions, batch_size=64, nb_epoch=40, verbose=2, validation_split=0.2)
    filename = open("pltdataforpreQ" + ".pickle","wb")
    pickle.dump(history.history, filename)
    filename.close()
    model.save("modelQ.h5")



def build_model(args):
    """
    Modified NVIDIA model
    """
    image1 = Input(shape=(84, 84, 4), name='image1')
    state = Input(shape=[35], name='state')
    c1 = Conv2D(32, 8, 8, activation='relu', subsample=(4,4))(image1)
    c2 = Conv2D(64, 4, 4, activation='relu', subsample=(2,2))(c1)
    c3 = Conv2D(64, 3, 3, activation='relu', subsample=(1,1))(c2)
    c4 = Conv2D(64, 3, 3, activation='relu', subsample=(1,1))(c3)
    c5 = Flatten()(c4)
    c6 = Dense(512, activation='relu')(c5)

    h0 = Dense(300, activation='relu')(state)
    h1 = Dense(600, activation='relu')(h0)
    conc = concatenate([c6, h1])
    out = Dense(moves, activation='linear', name='out')(conc)
    model = Model(inputs=[image1,state], outputs=out)
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate), metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file='model.png')
    return model




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

    model = build_model(args)
    history = load_data(args, model)




if __name__ == '__main__':
    main()
