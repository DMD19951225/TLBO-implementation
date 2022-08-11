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
# display the original images.
real = "./real_and_fake_face/training_real/"
fake = "./real_and_fake_face/training_fake/"

real_path = os.listdir(real)
fake_path = os.listdir(fake)

def load_img(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (200, 200))
    return image[...,::-1]

fig = plt.figure(figsize=(5, 5))

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(load_img(real + real_path[i]), cmap='gray')
    plt.title("real face")
    plt.axis('off')

plt.show()

fig = plt.figure(figsize=(5, 5))

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(load_img(fake + fake_path[i]), cmap='gray')
    plt.title("fake face")
    plt.axis('off')

plt.show()

# ************************************************************************************
def read_images():
    file_name = "real_and_fake_face"
    data_directory = "" + file_name

    data = np.zeros((2041, 2500))
    labels = np.zeros((2041, 1))

    i = 0
    j = 0
    for name in os.listdir(data_directory):
        folderPath = os.path.join(data_directory, name)
        for ImageName in os.listdir(folderPath):
            Image_path = os.path.join(folderPath, ImageName)

            img = cv2.imread(Image_path)
            img = cv2.resize(img, (50, 50))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            data[i, :] = img.flatten()
            labels[i] = j
            i += 1
        j += 1
    return data, labels

data, labels = read_images()

# def fitness(position):
#     fitness_value = 0.0
#     for i in range(len(position)):
#         xi = position[i]
#         fitness_value += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
#     return fitness_value
#
# # Teaching learning based optimization
# # data = pd.read_csv("Obfuscated-MalMem2022.csv")  # importing the csv file from folder
# # X = data.iloc[:, 2:56]  # select original features data
# data = pd.DataFrame(data)
# dim = 2500  # dimension of sample
# n = 2041  # number of samples
# max_iter = 5  # maximum repeat number
# print(data.shape[0])
# # TLBO algorithm implementation
# for iter in range(max_iter):
#     # teaching phase
#     f = []  # create the fitness list
#     for i in range(n):  # check each sample
#         f.append(fitness(data.iloc[i, :]))
#     pos = np.argmin(f)  # position of minimum fitness
#     X_teacher = data.iloc[pos, :]  # select the teacher as learner with minimum fitness
#     X_mean = data.mean(0)  # mean of total samples
#     TF = random.randint(1, 2)  # TF is the teaching factor and is either 1 or 2(chosen randomly)
#     r = random.random()  # random number range from 0 to 1
#     for i in range(n):
#         Xnew = data.iloc[i, :] + r * (X_teacher - TF * X_mean)
#         fnew = fitness(Xnew)
#         if fnew < fitness(data.iloc[i, :]):
#             data.iloc[i, :] = Xnew
#
#     # learning phase
#     p = random.randint(1, n-1)
#     X_partner = data.iloc[p, :]  # randomly chosen sample
#     if fitness(data.iloc[i, :]) < fitness(X_partner):
#         Xnew = data.iloc[i, :] + r * (data.iloc[i, :] - X_partner)
#     else:
#         Xnew = data.iloc[i, :] - r * (data.iloc[i, :] - X_partner)
#
#     fnew = fitness(Xnew)
#     if fnew < fitness(data.iloc[i, :]):
#         data.iloc[i, :] = Xnew
#
#     # chaotic phase
#     xi2 = np.random.random()
#     mu = 4
#     for i in range(n-1):
#         xi2 = mu * xi2 * (1 - xi2)
#         Xnew = data.iloc[i, :] + xi2
#         if fitness(Xnew) < fitness(data.iloc[i, :]):
#             data.iloc[i, :] = Xnew
# # ************************************************************************************
data = np.array(data)
print(data.shape)
n_rows = 50
n_cols = 50
n_channels = 1
X_Train, X_Test, y_Train, y_Test = train_test_split(
    data, labels, test_size=0.5, random_state=42)
X_train = np.reshape(data, newshape=(data.shape[0], n_rows, n_cols, n_channels))
X_test = np.reshape(X_Test, newshape=(X_Test.shape[0], n_rows, n_cols, n_channels))
y_train = to_categorical(labels, num_classes=2)
y_test = to_categorical(y_Test, num_classes=2)

'''Initializing the Convolutional Neural Network'''
classifier = Sequential()

''' STEP--1 Convolution
# Adding the first layer of CNN
# we are using the format (64,64,3) because we are using TensorFlow backend
# It means 3 matrix of size (64X64) pixels representing Red, Green and Blue components of pixels
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
import time

# Measuring the time taken by the model to train
StartTime = time.time()

# Starting the model training
classifier.fit(
    X_train, y_train,
    steps_per_epoch=30,
    epochs=30,
    validation_data=(X_test, y_test),
    validation_steps=10)

# '''########### Making single predictions ###########'''
# import numpy as np
# from keras.preprocessing import image
#
# ImagePath = '/Users/farukh/Python Case Studies/Face Images/Final Testing Images/face4/3face4.jpg'
# test_image = image.load_img(ImagePath, target_size=(64, 64))
# test_image = image.img_to_array(test_image)
#
# test_image = np.expand_dims(test_image, axis=0)

classifier.predict(X_test, verbose=0)

