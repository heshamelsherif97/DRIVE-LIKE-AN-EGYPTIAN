from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy
import pickle

history = None
with (open("pltdataforbehavioralOvbs.pickle", "rb")) as openfile:
    while True:
        try:
            history = pickle.load(openfile)
        except EOFError:
            break

plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
