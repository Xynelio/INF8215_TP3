"""
Team:
<<<<< TEAM NAME >>>>>
Authors:
<<<<< NOM COMPLET #1 - MATRICULE #1 >>>>>
<<<<< NOM COMPLET #2 - MATRICULE #2 >>>>>
"""

from wine_testers import WineTester
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np


class MyWineTester(WineTester):
    def __init__(self):
        # TODO: initialiser votre modèle ici:
        model = models.Sequential()
        model.add(layers.LayerNormalization())
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(32, activation="relu"))
        model.add(layers.Dense(16, activation="relu"))
        model.add(layers.Dense(10))
        model.add(layers.Softmax(0))
        # model.summary()
        self.model = model

    def train(self, X_train, y_train):
        """
        train the current model on train_data
        :param X_train: 2D array of data points.
                each line is a different example.
                each column is a different feature.
                the first column is the example ID.
        :param y_train: 2D array of labels.
                each line is a different example.
                the first column is the example ID.
                the second column is the example label.
        """
        # TODO: entrainer un modèle sur X_train & y_train
        features = []
        labels = []
        for line in X_train:
            if line[1] == "white":
                line[1] = 0
            elif line[1] == "red":
                line[1] = 1
            features.append(list(map(float, line[1:])))
        for line in y_train:
            labels.append(line[1])
        features = np.array(features)
        for i in range(np.shape(features)[1]):
            min_val = np.min(features[:, i])
            max_val = np.max(features[:, i])
            features[:, i] = (features[:, i] - min_val) / (max_val - min_val)

        features_tensor = tf.convert_to_tensor(features)
        labels_tensor = tf.convert_to_tensor(labels)

        self.model.compile(optimizer="adam",
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False))  # ,metrics=["accuracy"])
        history = self.model.fit(features_tensor[:-100], labels_tensor[:-100], epochs=50,
                       validation_data=(features_tensor[-100:], labels_tensor[-100:]))
        print(history)

    def predict(self, X_data):
        """
        predict the labels of the test_data with the current model
        and return a list of predictions of this form:
        [
            [<ID>, <prediction>],
            [<ID>, <prediction>],
            [<ID>, <prediction>],
            ...
        ]
        :param X_data: 2D array of data points.
                each line is a different example.
                each column is a different feature.
                the first column is the example ID.
        :return: a 2D list of predictions with 2 columns: ID and prediction
        """
        # TODO: make predictions on X_data and return them

        features = []
        labels = []
        for line in X_data:
            if line[1] == "white":
                line[1] = 0
            elif line[1] == "red":
                line[1] = 1
            features.append(list(map(float, line[1:])))

        features = np.array(features)
        for i in range(np.shape(features)[1]):
            min_val = np.min(features[:, i])
            max_val = np.max(features[:, i])
            features[:, i] = (features[:, i] - min_val) / (max_val - min_val)

        features_tensor = tf.convert_to_tensor(features)
        label_tensor = tf.argmax(self.model(features_tensor), axis=1)
        # print(tf.squeeze(label_tensor))
        for id, prediction in enumerate((tf.squeeze(label_tensor)).numpy().tolist()):
            labels.append([id, prediction])
        return labels
