import numpy as np
import cv2
import math
import random
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

warnings.simplefilter("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Create 2 virtual GPUs with 1GB memory each
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
         tf.config.LogicalDeviceConfiguration(memory_limit=1024),
         tf.config.LogicalDeviceConfiguration(memory_limit=1024),
         tf.config.LogicalDeviceConfiguration(memory_limit=1024),])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


# ************************************************************************************
def read_images():
    file_name = "real_and_fake_face"  # folder name selection of dataset
    data_directory = "" + file_name  # URL of dataset

    data = np.zeros((6123, 2500))  # to make grayscale image
    labels = np.zeros((6123, 1))

    i = 0
    j = 0
    for name in os.listdir(data_directory):
        folderPath = os.path.join(data_directory, name)
        for ImageName in os.listdir(folderPath):
            Image_path = os.path.join(folderPath, ImageName)
            # fig = plt.figure(figsize=(1,1))
            img = cv2.imread(Image_path)
            img = cv2.resize(img, (50, 50))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # plt.imshow(img, cmap='gray')
            data[i, :] = img.flatten()
            labels[i] = j
            i += 1
            plt.show()
        j += 1
    return data, labels

data, labels = read_images()

def fitness(position):
    fitness_value = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitness_value += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
    return fitness_value

data2 = pd.DataFrame(data)
dim = 2500  # dimension of sample
n = 6123  # number of samples
max_iter = 5  # maximum repeat number

#TLBO algorithm implementation
for iter in range(max_iter):
    # teaching phase
    f = []  # create the fitness list
    for i in range(n):  # check each sample
        f.append(fitness(data2.iloc[i, :]))
    pos = np.argmin(f)  # position of minimum fitness
    X_teacher = data2.iloc[pos, :]  # select the teacher as learner with minimum fitness
    X_mean = data2.mean(0)  # mean of total samples
    TF = random.randint(1, 2)  # TF is the teaching factor and is either 1 or 2(chosen randomly)
    r = random.random()  # random number range from 0 to 1
    for i in range(n):
        Xnew = data2.iloc[i, :] + r * (X_teacher - TF * X_mean)
        fnew = fitness(Xnew)
        if fnew < fitness(data2.iloc[i, :]):
            data2.iloc[i, :] = Xnew

    # learning phase
    p = random.randint(1, n-1)
    X_partner = data2.iloc[p, :]  # randomly chosen sample
    if fitness(data2.iloc[i, :]) < fitness(X_partner):
        Xnew = data2.iloc[i, :] + r * (data2.iloc[i, :] - X_partner)
    else:
        Xnew = data2.iloc[i, :] - r * (data2.iloc[i, :] - X_partner)

    fnew = fitness(Xnew)
    if fnew < fitness(data2.iloc[i, :]):
        data2.iloc[i, :] = Xnew

    # chaotic phase
    xi2 = np.random.random()
    mu = 4
    for i in range(n-1):
        xi2 = mu * xi2 * (1 - xi2)
        Xnew = data2.iloc[i, :] + xi2
        if fitness(Xnew) < fitness(data2.iloc[i, :]):
            data2.iloc[i, :] = Xnew
# ************************************************************************************
data2 = np.array(data2)
n_rows = 50
n_cols = 50
n_channels = 1

# split the data as train and test dataset
X_Train, X_Test, y_Train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42)
X_Train2, X_Test2, y_Train2, y_test2 = train_test_split(
    data2, labels, test_size=0.2, random_state=42)
X_train = np.reshape(X_Train, newshape=(X_Train.shape[0], n_rows, n_cols, n_channels))
X_test = np.reshape(X_Test, newshape=(X_Test.shape[0], n_rows, n_cols, n_channels))
y_train = to_categorical(y_Train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)
X_train2 = np.reshape(X_Train2, newshape=(X_Train2.shape[0], n_rows, n_cols, n_channels))
X_test2 = np.reshape(X_Test2, newshape=(X_Test2.shape[0], n_rows, n_cols, n_channels))
y_train2 = to_categorical(y_Train2, num_classes=2)
y_test2 = to_categorical(y_test2, num_classes=2)

'''Initializing the Convolutional Neural Network'''
classifier = Sequential()

'''
classifier.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(50, 50, 1), activation='relu'))

'''# STEP--2 MAX Pooling'''
classifier.add(MaxPool2D(pool_size=(2, 2)))

'''############## ADDITIONAL LAYER of CONVOLUTION for better accuracy #################'''
classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))

classifier.add(MaxPool2D(pool_size=(2, 2)))

'''# STEP--3 FLattening'''
classifier.add(Flatten())

'''# STEP--4 Fully Connected Neural Network'''
classifier.add(Dense(64, activation='relu'))

classifier.add(Dense(2, activation='softmax'))

'''# Compiling the CNN'''
# classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
###########################################################
# Starting the model training
classifier.fit(
    X_train, y_train,
    steps_per_epoch=40,
    epochs=50,
    validation_data=(X_test, y_test),
    validation_steps=10)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
predicted_labels = classifier.predict(X_test, verbose=0)
predicted_labels = np.argmax(np.round(predicted_labels), axis=1)
target_names = ['fake face', 'real face']
y_test = np.array(y_test[:, 1])

print("----------------------------------------------")
print("classification information without applying the Chaotic TLBO")
con = confusion_matrix(y_test, predicted_labels)
confusion_mat = pd.DataFrame(data = con, index =["fake face", "real face"],
                  columns =["fake face", "real face"])
print(confusion_mat)
print(classification_report(y_test, predicted_labels, target_names=target_names))


classifier2 = Sequential()

''' STEP--1 Convolution
'''
classifier2.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(50, 50, 1), activation='relu'))

'''# STEP--2 MAX Pooling'''
classifier2.add(MaxPool2D(pool_size=(2, 2)))

'''############## ADDITIONAL LAYER of CONVOLUTION for better accuracy #################'''
classifier2.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))

classifier2.add(MaxPool2D(pool_size=(2, 2)))

'''# STEP--3 FLattening'''
classifier2.add(Flatten())

'''# STEP--4 Fully Connected Neural Network'''
classifier2.add(Dense(64, activation='relu'))

classifier2.add(Dense(2, activation='softmax'))

'''# Compiling the CNN'''
classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
classifier2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
classifier2.fit(
    X_train2, y_train2,
    steps_per_epoch=30,
    epochs=50,
    validation_data=(X_test2, y_test2),
    validation_steps=10
)
predicted_labels2 = classifier2.predict(X_test2, verbose=0)
predicted_labels2 = np.argmax(np.round(predicted_labels2), axis=1)
target_names = ['fake face', 'real face']
y_test2 = np.array(y_test2[:, 1])
print("----------------------------------------------")
print("classification information after applying the Chaotic TLBO")
con2 = confusion_matrix(y_test2, predicted_labels2)
confusion_mat2 = pd.DataFrame(data = con2, index =["fake face", "real face"],
                  columns =["fake face", "real face"])
print(confusion_mat2)
print(classification_report(y_test2, predicted_labels2, target_names=target_names))
