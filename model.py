import os
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

def load_data():
    train_dir = Path('traffic_Data/DATA')
    test_dir = Path('traffic_Data/TEST')

    train_filepaths = list(train_dir.glob(r'*/.png'))
    test_filepaths = list(test_dir.glob(r'*/.png'))

    train_labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], train_filepaths))
    test_labels = list(map(lambda x: str(int(os.path.split(x)[1].split('_')[0])), test_filepaths))

    train_filepaths = pd.Series(train_filepaths, name='Filepath').astype(str)
    test_filepaths = pd.Series(test_filepaths, name='Filepath').astype(str)
    train_labels = pd.Series(train_labels, name='Label')
    test_labels = pd.Series(test_labels, name='Label')

    train_df = pd.concat([train_filepaths, train_labels], axis=1).sample(frac=1, random_state=1).reset_index(drop=True)
    test_df = pd.concat([test_filepaths, test_labels], axis=1).sample(frac=1, random_state=1).reset_index(drop=True)

    return train_df, test_df

def get_generators(train_df, test_df):
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    ) 

    train_images = train_generator.flow_from_dataframe(
        dataframe = train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        suffle=True,
        seed=1,
        subset='training'
    )

    val_images = train_generator.flow_from_dataframe(
        dataframe = train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=1,
        subset='validation'
    )

    test_images = test_generator.flow_from_dataframe(
        dataframe = test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        classes=train_images.class_indices,
        batch_size=32,
        shuffle=False
    )

    return train_images, val_images, test_images

class Model:
    def _init_(self):
        self.model = load_model('model.h5')
        self.train_df, self.test_df = load_data()
        self.train_images, self.val_images, self.test_images = get_generators(self.train_df, self.test_df)

    def summary(self):
        self.model.summary()

    def predict(self, datafile):
        img = cv2.imread(datafile) 
        img = cv2.resize(img, (224,224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255
        img = img[np.newaxis, ...]
        return np.argmax(self.model.predict(img))

    def test(self):
        results = self.model.evaluate(self.test_images, verbose=0)
        print("Test Accuracy: {:.2f}%".format(results[1] * 100))