from os import listdir
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import random
from keras.models import  load_model
import sys

class test:
    def __init__(self, video, weight_path):
        self.video = video
        self.class_name = ['0','100000','200000','500000']
        self.weight_path = weight_path
        self.my_model = 0

    def __get_model(self):
        model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

        # Freeze the convolution
        for layer in model_vgg16_conv.layers:
            layer.trainable = False

        # Generating the input
        input = Input(shape=(128, 128, 3), name='image_input')
        output_vgg16_conv = model_vgg16_conv(input)

        # Edit the output layer
        x = Flatten(name='flatten')(output_vgg16_conv)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dropout(0.5)(x)
        x = Dense(4, activation='softmax', name='predictions')(x)

        # Compile model
        model = Model(inputs=input, outputs=x)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def load_model(self):
        self.my_model = self.__get_model()
        self.my_model.load_weights(self.weight_path)

    def test_model(self):
        while(1):
            ret, frame = self.video.read()

            if not ret:
                pass

            frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
            #Resize
            image = frame.copy()
            image = cv2.resize(image, dsize=(128, 128))
            image = image.astype('float') * 1. / 255
            #Convert to tensor
            image = np.expand_dims(image, axis=0)

            # Predict
            predict = self.my_model.predict(image)
            print("This picture is: ", self.class_name[np.argmax(predict[0])], (predict[0]))
            print(np.max(predict[0], axis=0))
            if (np.max(predict) >= 0.8) and (np.argmax(predict[0]) != 0):
                # Show image
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (50, 50)
                fontScale = 1.5
                color = (0, 255, 0)
                thickness = 2
                cv2.putText(frame, self.class_name[np.argmax(predict)], org, font,
                            fontScale, color, thickness, cv2.LINE_AA)

            cv2.imshow("Picture", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        #Realease the video
        self.video.release()
        cv2.destroyAllWindows()

    def __call__(self, *args, **kwargs):
        self.load_model()
        self.test_model()



if __name__ == "__main__":
    #Generate the video and weight path
    camera_id = 0
    video = cv2.VideoCapture(camera_id)
    weight_path = "weights-37-0.99.hdf5"


    #Testing the model
    testing = test(video, weight_path=weight_path)
    testing()









































