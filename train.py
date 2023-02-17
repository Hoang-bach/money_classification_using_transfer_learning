import os
import cv2
import numpy as np
import pickle
from os import listdir
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer


class train:
    def __int__(self):
        self.raw_folder = 'dataset/'
        self.labels = []
        self.pixels = []
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []


    def save_data(self):
        print("Starting")

        #Read through folder in raw folder
        for folder in listdir(self.raw_folder):
            if folder != '.DS_Store':
                print("Folder=", folder)
                #Read through file in folder
                for file in listdir(self.raw_folder + folder):
                    if file != ".DS_Store":
                        print("Folder=", file)
                        self.pixels.append(cv2.resize(cv2.imread(self.raw_folder + folder + "/" + file), dsize=(128, 128)))
                        self.labels.append(folder)
        self.pixels = np.array(self.pixels)
        self.labels = np.array(self.labels)

        #Encode the label
        encoder = LabelBinarizer()
        self.labels = encoder.fit_transform(self.labels)

        #Save the label to one file
        file = open('pic.data', 'wb')
        pickle.dump((self.pixels, self.labels), file)
        file.close()
        return

    def load_data(self):
        #Load the label file
        file = open('pic.data', 'rb')
        (self.pixels, self.labels) = pickle.load(file)
        file.close()

        #Generate data for training
        X, y = self.pixels, self.labels
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=100)

    def __get_model(self):
        model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

        #Freeze the convolution
        for layer in model_vgg16_conv.layers:
            layer.trainable = False

        #Generating the input
        input = Input(shape=(128, 128, 3), name='image_input')
        output_vgg16_conv = model_vgg16_conv(input)

        #Edit the output layer
        x = Flatten(name='flatten')(output_vgg16_conv)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dropout(0.5)(x)
        x = Dense(4, activation='softmax', name='predictions')(x)

        #Compile model
        my_model = Model(inputs=input, outputs=x)
        my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return my_model

    def train(self):
        my_model = self.__get_model()

        #Generate checkpoint
        filepath = "weights-{epoch:02d}-{val_accuracy:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        #Augmentation for the data
        augment = ImageDataGenerator(rotation_range=20, zoom_range=0.1,
                                    rescale=1./255,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    horizontal_flip=True,
                                    brightness_range=[0.2,1.5], fill_mode="nearest")

        my_model.fit_generator(augment.flow((self.X_train, self.y_train), batch_size=64),
                               epochs=50,
                               validation_data=augment.flow(self.X_test, self.y_test, batch_size=64),
                               callbacks=callbacks_list)
        my_model.save("my_model.h5")

    def __call__(self, *args, **kwargs):

        #Load data from the pic.data file
        self.load_data()

        #Training
        self.train()

if __name__ == "__main__":
    training = train()

    #Save dataset to pic.data file
    # training.save_data()

    #Start training
    training()



