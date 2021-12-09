"""
Team:
La bonne Vinasse

Authors:
Eloi DUSSY LACHAUD - 2098721
Alexandre VERDET - 2164847
"""

from wine_testers import WineTester
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import svm


def preprocess(X_train, y_train):
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

    # features = SelectKBest(chi2, k=10).fit_transform(features, labels)

    # Manual scaling
    # for i in range(np.shape(features)[1]):
    #     min_val = np.min(features[:, i])
    #     max_val = np.max(features[:, i])
    #     features[:, i] = (features[:, i] - min_val) / (max_val - min_val)

    return features, labels


class MyWineTester(WineTester):
    def __init__(self):
        parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10], 'gamma': [0.5, 2]}
        self.clf = make_pipeline(StandardScaler(), svm.SVC(C=1.3, kernel="rbf", gamma=0.65))
        # self.clf = make_pipeline(StandardScaler(), GridSearchCV(svm.SVC(), parameters))
        print(self.clf)

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

        X_train, y_train = preprocess(X_train, y_train)

        # Split dataset in training and testing subset
        # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=0)
        # Training
        self.clf.fit(X_train, y_train)

        # Scoring to evaluate hyperparameters, using cross validation and grid search :
        # scores = cross_val_score(self.clf, X_train, y_train, cv=5)
        # print(scores)
        # print(sorted(self.clf.named_steps["gridsearchcv"].best_estimator_))
        # print(self.clf.get_params())
        # print(self.clf.score(X_test, y_test))

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

        X_data, _ = preprocess(X_data, [])
        labels = []

        preds = np.round(self.clf.predict(X_data).tolist())
        for id, prediction in enumerate(preds):
            labels.append([id, prediction])
        return labels

