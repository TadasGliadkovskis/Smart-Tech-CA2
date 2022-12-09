import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import random
import requests
from PIL import Image
import cv2
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import pickle
import pandas as pd
import cv2

num_classes = 43


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


def lenet_model():
    model = Sequential
    model.add(Conv2D(30, (5, 5), input_shape=(32, 32, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def modified_model():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5)) Reduce time compiling model
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def show_images_data():
    num_of_samples = []
    cols = 3
    fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(50, 50))
    fig.tight_layout()
    for i in range(cols):
        for j, row in sign_names.iterrows():
            x_selected = X_train[y_train == j]
            axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap("gray"))
            axs[j][i].axis("off")
            if i == 2:
                num_of_samples.append(len(x_selected))
                # axs[j][i].set_title(str(j)+'-'+row['SignName']) puts name of img

    plt.figure(figsize=(12, 4))
    plt.bar(range(0, num_classes), num_of_samples)
    plt.title("Distribution of training data")
    plt.ytitle("Number of images")
    plt.xtitle("Class Number")
    plt.show()

with open('german-traffic-signs/train.p', 'rb') as f:
    train_data = pickle.load(f)

with open('german-traffic-signs/valid.p', 'rb') as f:
    val_data = pickle.load(f)

with open('german-traffic-signs/test.p', 'rb') as f:
    test_data = pickle.load(f)

# print(type(train_data))

X_train, y_train = train_data['features'], train_data['labels']
X_val, y_val = val_data['features'], val_data['labels']
X_test, y_test = test_data['features'], test_data['labels']

assert (X_train.shape[0] == y_train.shape[0]), "The number of training images is not equal to the number of labels"
assert (X_val.shape[0] == y_val.shape[0]), "The number of validation images is not equal to the number of labels"
assert (X_test.shape[0] == y_test.shape[0]), "The number of test images is not equal to the number of labels"
# assert (X_train.shape[1:] == y_train.shape[1:]), "The image dimensions on the training images are not 32x32x3"
# assert (X_val.shape[1:] == y_val.shape[1:]), "The image dimensions on the validation images are not  32x32x3"
# assert (X_test.shape[1:] == y_test.shape[1:]), "The image dimensions on the test images are not 32x32x3"

sign_names = pd.read_csv('german-traffic-signs/signnames.csv')

plt.imshow(X_train[1000])
plt.axis('off')
plt.show()

X_train = np.array(list(map(preprocessing, X_train)))
X_val = np.array(list(map(preprocessing, X_val)))
X_test = np.array(list(map(preprocessing, X_test)))

plt.imshow(X_train[random.randint(0, len(X_train) - 1)])
plt.axis('off')
plt.show()
print(X_train.shape)

# Convolutional neural network needs a depth field '1'
X_train = X_train.reshape(X_train.shape[0], 32, 32, 1)
X_test = X_test.reshape(X_test.shape[0], 32, 32, 1)
X_val = X_val.reshape(X_val.shape[0], 32, 32, 1)

datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
datagen.fit(X_train)

# batches = datagen.flow(X_train, y_train, batch_size=20) creates a batch of 20 pics and shows them how theyve modified
# X_batch, y_batch = next(batches)
# fig, axs = plt.subplots(1, 20, figsize=(20,5))
# for i in range(20):
#     axs[i].imshow(X_batch[i].reshape(32, 32))
#     axs[i].axis("off")

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
y_val = to_categorical(y_val, num_classes)

model = modified_model()
history = model.fit(X_train, y_train, batch_size=50, steps_per_epoch=X_train.shape[0]/50, epochs=20, validation_data=(X_val, y_val), verbose=1, shuffle=1)



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')
score = model.evaluate
plt.show()

score = model.evaluate(X_test, y_test, verbose=0)
print('Test score: ', score[0])
print('Test accuracy: ', score[1])

url = 'https://www.123rf.com/photo_101026938_speed-limit-sign-70-in-rural-area.html?vti=n71gwnqi5lazzntk8n-2-108'

req = requests.get(url, stream=True)
img = Image.open(req.raw)
plt.show(img, cmap=plt.get_cmap('gray'))
plt.show()

img = np.asarray(img)
img = cv2.resize(img, (32, 32))
img = preprocessing(img)
img = img.reshape(1, 32, 32, 1)
print("Predicted Sign: "+ str(np.argmax(model.predict(img), axis=1)))
