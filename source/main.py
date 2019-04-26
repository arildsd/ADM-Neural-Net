from __future__ import division

import os

from keras.utils import np_utils

from pre_processing import pre_process
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten, Reshape, Dropout
import time
import copy
import matplotlib
import matplotlib.pyplot as plt


print('Using Keras version', keras.__version__)
from keras.datasets import cifar100

#print('Using tenserflow version', tensorflow.__version__)

OUTPUT_FOLDER_NAME = "1x4layer"
ITERATIONS = 20
EPOCHS = 1000

df = pre_process()
train, validation = train_test_split(df, test_size=0.25, random_state=1)

train_x = train.drop(columns="student_performance")
train_y = train["student_performance"]

validation_x = validation.drop(columns="student_performance")
validation_y = validation["student_performance"]


# Adapt the labels to the one-hot vector syntax required by the softmax
train_y = np_utils.to_categorical(train_y, 3)
validation_y = np_utils.to_categorical(validation_y, 3)
#Model visualization
#We can plot the model by using the ```plot_model``` function. We need to install *pydot, graphviz and pydot-ng*.
#from keras.util import plot_model
#plot_model(nn, to_file='nn.png', show_shapes=true)

#Compile the NN

start_time = time.time()
summed_val_acc_history = np.array([0.0]*EPOCHS)
summed_train_acc_history = np.array([0.0]*EPOCHS)
max_val_acc = []
max_train_acc = []
for i in range(ITERATIONS):
    nn = Sequential()
    nn.add(Dense(4, activation='relu', input_shape=(train_x.shape[1],)))
    nn.add(Dense(3, activation='softmax'))

    sgd = keras.optimizers.SGD(lr=0.05)
    nn.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    history = nn.fit(train_x, train_y, epochs=EPOCHS, validation_data=(validation_x, validation_y),
                    verbose=0, batch_size=1024)
    print(f"Iteration: {i}\tBest training accuracy: %f\tBest validation accuracy: %f" %
          (max(history.history["acc"]), max(history.history["val_acc"])))
    max_val_acc.append(max(history.history["val_acc"]))
    max_train_acc.append(max(history.history["acc"]))
    summed_val_acc_history += np.array(history.history["val_acc"])
    summed_train_acc_history += np.array(history.history["acc"])

avr_val_acc_history = summed_val_acc_history/ITERATIONS
avr_train_acc_history = summed_train_acc_history/ITERATIONS

duration = time.time()-start_time
output_string = f"Duration: {duration}\nBest average training accuracy: {sum(max_train_acc)/ITERATIONS}\n" \
    f"Best average validation accuracy: {sum(max_val_acc)/ITERATIONS}\n" \
    f"Average test accuracy: {list(avr_train_acc_history).__str__()}\n" \
    f"Average validation accuracy: {list(avr_val_acc_history).__str__()}"
print("\n")
print(output_string)
nn.summary()
##Store Plots

#Accuracy plot
os.makedirs(f"../output/{OUTPUT_FOLDER_NAME}")
f = open(f"../output/{OUTPUT_FOLDER_NAME}/log.txt", "w")
f.write(output_string)
f.close()

plt.plot(avr_train_acc_history)
plt.plot(avr_val_acc_history)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(f"../output/{OUTPUT_FOLDER_NAME}/training_validation_accuracy.png")
plt.show()
plt.close()
