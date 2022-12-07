import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import random
from PIL import Image
import cv2
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import pandas as pd
import cv2
import ntpath
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail


def load_steering_img(datadir, data):
    image_path = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        centre, left, right = indexed_data[0], indexed_data[1], indexed_data[2]  # leaving out left and right for CA2
        image_path.append(os.path.join(datadir, centre.strip()))
        steering.append(float(indexed_data[3]))
    image_paths = np.array(image_path)
    steerings = np.array(steering)
    return image_paths, steerings


def preprocess_img(img):
    img = mpimg.imread(img)
    img = img[60:135, :, :] #X, Y RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img (200, 66))
    img = img/255
    return img

def nvidia_model():
    model = Sequential()
    model.add(Convolution2D(24, (5, 5), (2,2),input_shape=(66,200,3), activation='relu'))
    model.add(Convolution2D(36, (5, 5), (2,2), activation='relu'))
    model.add(Convolution2D(48, (5, 5), (2,2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(Convolution2D(48, (3, 3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='relu'))
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=optimizer)
    return model



datadir = "C:"
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names=columns)
pd.set_option('max_columns', 7)

data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)

num_bins = 25
samples_per_bin = 200
hist, bins = np.histogram(data['steering'], num_bins)
print(bins[:-1])
print(bins[1:])
centre = (bins[:-1] + bins[1:] * 0.5)
plt.bar(centre, hist, width=0.1)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
plt.show()

remove_list = []
print('Total data' + len(data))

for j in range(num_bins):
    list_ = []
    for i in range(len(data['steering'])):
        if bins[j] <= data['steering'][i] <= bins[j + 1]:
            list_.append(i)
    list_ = shuffle(list_)
    list_ = list_[samples_per_bin:]
    remove_list.extend(list_)  # .extend is memory efficient

print("Remove: ", len(remove_list))
data.drop(data.index[remove_list], inplace=True)
print("Remaining data", len(data))

hist, bins = np.histogram(data['steering'], num_bins)
centre = (bins[:-1] + bins[1:] * 0.5)
plt.bar(centre, hist, width=0.1)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
plt.show()

image_paths, steerings = load_steering_img(datadir + '/IMG', data)
X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)
print(f"Training Samples { len(X_train)}\n Validation Samples {len(X_valid)}\n")

fig, axis = plt.subplots(1,2, figsize=(12,4))
axis[0].hist(y_train, bins=num_bins, width=0.05, color='blue')
axis[0].set_title("Training set")
axis[1].hist(y_valid, bins=num_bins, width=0.05, color='red')
axis[1].set_title("Validation Set")
plt.show()

image = image_paths[100]
original_image = mpimg.imread(image)
preprocessed_img = preprocess_img(image)

fig, axis = plt.subplots(1,2, figsize=(15,10))
fig.tight_layout()
axis[0].imshow(original_image)
axis[0].set_title("Original Image")
axis[1].imshow(preprocessed_img)
axis[1].set_title("Preprocessed Image")
plt.show()

X_train = np.array(list(map(preprocess_img(), X_train)))
X_train = np.array(list(map(preprocess_img(), X_valid)))

plt.imshow(X_train[random.randint(0, len(X_train)-1)])
plt.axis('off')
plt.show()
print(X_train.shape)




