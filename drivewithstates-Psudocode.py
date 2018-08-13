import argparse
import base64
from datetime import datetime
import os
import shutil
import utilsQ
import socket
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from utilsQ import INPUT_SHAPE, batch_generator
from keras.utils.vis_utils import plot_model
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
from keras.layers import Dense, Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
import random
import time
import pickle
host <- "localhost"
port <- 4567
backlog <- 5
size <- 100000
s <- socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((host,port))
s.listen(backlog)
sio <- socketio.Server()
app <- Flask(__name__)
global model
prev_image_array <- None
FUNCTION s2b(s):
    """
    Converts a string to boolean value
    """
    s <- s.lower()
    RETURN s = 'true'
ENDFUNCTION

FUNCTION train(flag):
    plotinghistory <- []
    targetrew <- []
    loss <- []
    desiredtarget <- []
    lossperstep <- []
    speed_limit <- 20
    #For now we make the car accelerate, turn right AND turn left
    moves <- 4
    #learning rate (discount rate)
    learningRate <- 0.9
    #This is the exploration rate (epsilon)
    #Its better at first to let the model try everything
    epsilon <- 1.0
    #We don't want our model to stop exploring so we set a minimum epsilon
    epsilon_min <- 0.01
    #We also dont want our model to explore all the time therefore we want it
                                                              ENDFOR
    #to decay
    epsilon_decay <- 0.995
    #We want to store our data for replay/so our model can remember
                               ENDFOR
    memory <- []
    #The max amount of stuff we want to remember
    max_memory <- 50000
    #model <- Sequential()
    image1 <- Input(shape=(84, 84, 4), name='image1')
    state <- Input(shape=[35], name='state')
    c1 <- Conv2D(32, (8, 8), activation='relu', subsample=(4,4))(image1)
    c2 <- Conv2D(64, (4, 4), activation='relu', subsample=(2,2))(c1)
    c3 <- Conv2D(64, (3, 3), activation='relu', subsample=(1,1))(c2)
    c4 <- Conv2D(64, (3, 3), activation='relu', subsample=(1,1))(c3)
    c5 <- Flatten()(c4)
    c6 <- Dense(512, activation='relu')(c5)
    h0 <- Dense(300, activation='relu')(state)
    h1 <- Dense(600, activation='relu')(h0)
    conc <- concatenate([c6, h1])
    out <- Dense(moves, activation='linear', name='out')(conc)
    model <- Model(inputs=[image1,state], outputs=out)
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-6), metrics=['accuracy'])
    model.summary()
    plot_model(model, to_file='modelQ.png', show_shapes=True)
    IF(os.path.isfile("modelQstate.h5")):
        model <- load_model("modelQstate.h5")
    ENDIF
    OUTPUT "Waiting For Connection with Simulator"
    client,address <- s.accept()
    OUTPUT "Simulator connected"
    i <- 0
    #loop over the number of epochs (essentially the number of games)
    while True:
        #We set the game_over to false as the game is just starting
        game_over <- False
        #Getting first Image
        client.send("Send\n".encode('utf-8'))
        imageee <- str(client.recv(200000),"utf-8").rstrip('\r\n')
        input_img1 <- Image.open(BytesIO(base64.b64decode(imageee)))
        #Getting Second Image
        client.send("Send\n".encode('utf-8'))
        imageee <- str(client.recv(200000),"utf-8").rstrip('\r\n')
        input_img2 <- Image.open(BytesIO(base64.b64decode(imageee)))
        #Getting Third Image
        client.send("Send\n".encode('utf-8'))
        imageee <- str(client.recv(200000),"utf-8").rstrip('\r\n')
        input_img3 <- Image.open(BytesIO(base64.b64decode(imageee)))
        #Getting fourth Image
        client.send("Send\n".encode('utf-8'))
        imageee <- str(client.recv(200000),"utf-8").rstrip('\r\n')
        input_img4 <- Image.open(BytesIO(base64.b64decode(imageee)))
        client.send("state\n".encode('utf-8'))
        statestring <- str(client.recv(200000),"utf-8").rstrip('\r\n')
        state <- statestring.split(",")
        tmp_state <- [None] *35
        state1 <- []
        for x in range(0,35):
            IF state[x] = '':
                tmp_state[x] <- 0
            ELSE:
                tmp_state[x] <- (round(float(state[x]), 2))
            ENDIF
        ENDFOR
        state1 <- np.array(tmp_state)
        state1 <- state1.reshape((1, 35))
                # save frame
        IF args.image_folder != '':
            timestamp <- datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename1 <- os.path.join(args.image_folder, timestamp)
            timestamp <- datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename2 <- os.path.join(args.image_folder, timestamp)
            timestamp <- datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename3 <- os.path.join(args.image_folder, timestamp)
            timestamp <- datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename4 <- os.path.join(args.image_folder, timestamp)
            input_img1.save('{}.jpg'.format(image_filename1))
                                     ENDFOR
            input_img2.save('{}.jpg'.format(image_filename2))
                                     ENDFOR
            input_img3.save('{}.jpg'.format(image_filename3))
                                     ENDFOR
            input_img4.save('{}.jpg'.format(image_filename4))
        ENDIF
                                     ENDFOR
        try:
            input_img1 <- np.asarray(input_img1)       # from PIL image to numpy array
            input_img1 <- utilsQ.preprocess(input_img1) # apply the preprocessing
            input_img1 <- input_img1.reshape(1, 84, 84)
            input_img2 <- np.asarray(input_img2)       # from PIL image to numpy array
            input_img2 <- utilsQ.preprocess(input_img2) # apply the preprocessing
            input_img2 <- input_img2.reshape(1, 84, 84)
            input_img3 <- np.asarray(input_img3)       # from PIL image to numpy array
            input_img3 <- utilsQ.preprocess(input_img3) # apply the preprocessing
            input_img3 <- input_img3.reshape(1, 84, 84)
            input_img4 <- np.asarray(input_img4)       # from PIL image to numpy array
            input_img4 <- utilsQ.preprocess(input_img4) # apply the preprocessing
            input_img4 <- input_img4.reshape(1, 84, 84)
        except Exception as e:
            OUTPUT e
        input_img <- np.stack((input_img1, input_img2, input_img3, input_img4), axis <- 3)
        #We set the errors to false to begin with
        errors <- False
        #We set the reward to 0
        reward <- 0
        treward <- 0
        #While the game is not over we loop
        while game_over==False:
            #Np.random.rand() returns a number between 0 and 1
            #We check IF its smaller that our exploration factor
                      ENDIF
            IF np.random.rand() <= epsilon and flag = True:
                #if the random number is smaller than our exploration factor
                 ENDIF
                #We select a random action from our 3 actions
                action <- np.random.randint(0, moves, size=1)[0]
                OUTPUT "Random Action Done : "+ str(action)
            ELSE:
                #If it's not smaller than we predict an output by inputting our
                #4 stacked images
                #ouput is the probability of our 3 directions
                output <- model.predict([input_img, state1])
                #action is the index of the highest probability AND therefore
                                                                         ENDFOR
                #indicates which turn to take
                action <- np.argmax(output[0])
                OUTPUT "Predicted Action Done : "+ str(action)
            ENDIF
            client.send("Action".encode('utf-8') + str(action).encode('utf-8')+"\n".encode('utf-8'))
            response2 <- str(client.recv(200000),"utf-8").rstrip('\r\n')
            #Once we've performed our action we get the next frame
                           ENDFOR
            #We also check weather to reward the algorithm or not
            client.send("Send\n".encode('utf-8'))
            response3 <- str(client.recv(200000),"utf-8").rstrip('\r\n')
            input_next_img1 <- Image.open(BytesIO(base64.b64decode(response3)))
            client.send("Send\n".encode('utf-8'))
            response3 <- str(client.recv(200000),"utf-8").rstrip('\r\n')
            input_next_img2 <- Image.open(BytesIO(base64.b64decode(response3)))
            client.send("Send\n".encode('utf-8'))
            response3 <- str(client.recv(200000),"utf-8").rstrip('\r\n')
            input_next_img3 <- Image.open(BytesIO(base64.b64decode(response3)))
            client.send("Send\n".encode('utf-8'))
            response3 <- str(client.recv(200000),"utf-8").rstrip('\r\n')
            input_next_img4 <- Image.open(BytesIO(base64.b64decode(response3)))
            client.send("state\n".encode('utf-8'))
            statestring <- str(client.recv(200000),"utf-8").rstrip('\r\n')
            state <- statestring.split(",")
            tmp_state2 <- [None] *35
            state2 <- []
            for x in range(0,35):
                IF state[x] = '':
                    tmp_state2[x] <- 0
                ELSE:
                    tmp_state2[x] <- (round(float(state[x]), 2))
                ENDIF
            ENDFOR
            state2 <- np.array(tmp_state2)
            state2 <- state2.reshape((1, 35))
            client.send("crash\n".encode('utf-8'))
            response4 <- str(client.recv(200000),"utf-8").rstrip('\r\n')
            errors <- s2b(response4)
                    # save frame
            IF args.image_folder != '':
                timestamp <- datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
                image_filename1 <- os.path.join(args.image_folder, timestamp)
                timestamp <- datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
                image_filename2 <- os.path.join(args.image_folder, timestamp)
                timestamp <- datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
                image_filename3 <- os.path.join(args.image_folder, timestamp)
                timestamp <- datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
                image_filename4 <- os.path.join(args.image_folder, timestamp)
                input_next_img1.save('{}.jpg'.format(image_filename1))
                                              ENDFOR
                input_next_img2.save('{}.jpg'.format(image_filename2))
                                              ENDFOR
                input_next_img3.save('{}.jpg'.format(image_filename3))
                                              ENDFOR
                input_next_img4.save('{}.jpg'.format(image_filename4))
            ENDIF
                                              ENDFOR
            try:
                input_next_img1 <- np.asarray(input_next_img1)       # from PIL image to numpy array
                input_next_img1 <- utilsQ.preprocess(input_next_img1) # apply the preprocessing
                input_next_img1 <- input_next_img1.reshape(1, 84, 84)
                input_next_img2 <- np.asarray(input_next_img2)       # from PIL image to numpy array
                input_next_img2 <- utilsQ.preprocess(input_next_img2) # apply the preprocessing
                input_next_img2 <- input_next_img2.reshape(1, 84, 84)
                input_next_img3 <- np.asarray(input_next_img3)       # from PIL image to numpy array
                input_next_img3 <- utilsQ.preprocess(input_next_img3) # apply the preprocessing
                input_next_img3 <- input_next_img3.reshape(1, 84, 84)
                input_next_img4 <- np.asarray(input_next_img4)       # from PIL image to numpy array
                input_next_img4 <- utilsQ.preprocess(input_next_img4) # apply the preprocessing
                input_next_img4 <- input_next_img4.reshape(1, 84, 84)
            except Exception as e:
                OUTPUT e
            input_next_img <- np.stack((input_next_img1, input_next_img2, input_next_img3, input_next_img4), axis <- 3)
            speed <- tmp_state2[30]
            #If we detect lanes AND therefore no errors occur we reward the algorithm
                                         ENDFOR
            IF errors = False:
                reward <- 10
                rewardfactor <- (speed_limit - abs(speed_limit - speed))/speed_limit
                reward <- reward * rewardfactOR
                for e in range(0, 30):
                    reward -= (1/tmp_state2[e])
                ENDFOR
            #Else IF there we detect no lanes AND so there is an error we
                  ENDIF
            #say its game over
            ELSE:
                game_over <- True
                reward <- -10
                rewardfactor <- (speed_limit - abs(speed_limit - speed))/speed_limit
                reward <- reward * rewardfactOR
                for e in range(0, 30):
                    reward -= (1/tmp_state2[e])
            ENDIF
                ENDFOR
            treward += reward
            OUTPUT reward
            #Game over OR not we want to keep record of the steps the algo took
            #We first check IF the total memoery length is bigger than the max memory
                            ENDIF
            IF len(memory) >= max_memory:
                #If more memory then needed we delete the first ever element we added
                del memory[0]
            #We append it to our memory list
            ENDIF
            memory.append((input_img, state1, action, reward, input_next_img, state2, game_over))
            #Next we set our input_img to our latest data
            input_img <- input_next_img
            state1 <- state2
            IF game_over:
                # epochs <- epochs + 1
                OUTPUT "New Run, Total Reward: {0}, Epoch: {1}".format(treward, i)
                                                               ENDFOR
                plotinghistory.append(treward)
                i <- i + 1
            ENDIF
        #Once the game is over we want to train our algo with the data we just collected
        #We check IF our memory length is bigger than our batch size 
                  ENDIF
        ENDWHILE
        IF len(memory) > 32:
        #If so then we set the batch_size to 32
            batch_size <- 32
        ELSE:
        #Else we set our batch size to whatever is in the memory
            batch_size <- len(memory)
        #We are taking a random sample of 32 so not to overfit our algo
        ENDIF
        batch <- random.sample(memory, batch_size)
        #We itereate over every memory we've stored in that memory batch of 32
        OUTPUT "Learning From Previous Run"
        lossc <- 0
        count <- 0
        for input_img, state1, action, reward, input_next_img, state2, game_over in batch:
            #if in that memory our game was over then we set the target_reward equal to reward
             ENDIF
            target_reward <- reward
            #If our game was not over
            IF game_over = False:
            #This essentially is the bellman equation
            #expected long-term reward for a given action is equal to the
                                       ENDFOR
            #immediate reward from the current action combined with the expected
            #reward from the best future action taken at the following state.
            #The model isn't certain that for that specific action it will get the best reward
                                                       ENDIF
                                          ENDFOR
            #It's based on probability of the action, IF the probability of that action is in the
                                                      ENDIF
            #negatives then our future reward is going to be further decreased by our learning rate
            #This is just the model being cautious, as to not set an impossible reward target
            #If the reward is impossible then the algorithm might not converge
            #Converge as in a stable condition where it can play the game without messing up
                target_reward <- reward + learningRate * \
                np.amax(model.predict([input_next_img, state2])[0])
            #So from above we essentially know what is going to happen(input_next_img)
            #assuming the game wasn't over, the algorithm did well.
            #So we want the algorithm to perform the same, essentially we
                                            ENDFOR
            #persuade the algorithm to do what it did to get that reward
            #so we make the algorithm predict from the previous frame(input_img)
            #but we alter its prediction according to the action that got the highest
            #reward AND...
            ENDIF
            targetrew.append(target_reward)
            desired_target <- model.predict([input_img, state1])
            #we set that as the target_reward...
            desired_target[0][action] <- target_reward
            desiredtarget.append(desired_target)
            #So to make the algo perform the same, we associate the input_img with the
                                    ENDFOR
            #target we want AND we fit it
            history <- model.fit([input_img, state1], desired_target, epochs=1, verbose=0)
            lossc += history.history["loss"][0]
            lossperstep.append(history.history["loss"][0])
            OUTPUT history.history["loss"]
            count +=1
        #Finally we check IF our exploration factor is bigger than our minimum exploration
                          ENDIF
        #if so we decrease it by the decay to reduce exploration, we do this every game
         ENDIF
        ENDFOR
        loss.append(lossc/count)
        IF epsilon > epsilon_min:
            epsilon *= epsilon_decay
        ENDIF
        client.send("restart\n".encode('utf-8'))
        response5 <- str(client.recv(200000),"utf-8").rstrip('\r\n')
        OUTPUT "Saving Model"
        model.save("modelQstate.h5", overwrite=True)
        filename <- open("pltdataforQ" + ".pickle","wb")
                                ENDFOR
        pickle.dump(plotinghistory, filename)
        filename.close()
        filename <- open("pltdatafortargrew" + ".pickle","wb")
                                ENDFOR
        pickle.dump(targetrew, filename)
        filename.close()
        filename <- open("pltdataforloss" + ".pickle","wb")
                                ENDFOR
        pickle.dump(loss, filename)
        filename.close()
        filename <- open("pltdatafordesiredtarget" + ".pickle","wb")
                                ENDFOR
        pickle.dump(desiredtarget, filename)
        filename.close()
        filename <- open("pltdataforlossperstep" + ".pickle","wb")
                                ENDFOR
        pickle.dump(lossperstep, filename)
        filename.close()
ENDFUNCTION

    ENDWHILE
IF __name__ = '__main__':
    parser <- argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        ENDFUNCTION

        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args <- parser.parse_args()
    IF args.image_folder != '':
        OUTPUT "Creating image folder at {}".format(args.image_folder)
                                            ENDFOR
        IF not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        ELSE:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        ENDIF
        OUTPUT "RECORDING THIS RUN ..."
    ELSE:
        OUTPUT "NOT RECORDING THIS RUN ..."
    ENDIF
    x <- True
    train(x)
