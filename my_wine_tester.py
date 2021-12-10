"""
Team:
<<<<< TEAM NAME >>>>>
Authors:
# VERDET Alexandre - 2164847
# DUSSY LACHAUD Eloi - 2098721
"""

from wine_testers import WineTester
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class MyWineTester(WineTester):
    def __init__(self):
        # TODO: initialiser votre modèle ici:
        self.predictor = None
        self.column_names = ["id", "color", "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates",
                        "alcohol"]
        self.convert_data = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates",
                        "alcohol"]
        self.model_features = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates",
                        "alcohol", "color_white"]

        pass

    def remove_outliers(self, X_dtf, y_dtf):
        dropIndexes = X_dtf[X_dtf["free sulfur dioxide"] > 83.5].index
        dropIndexes.append(X_dtf[X_dtf["chlorides"] > 0.31].index)
        X_dtf.drop(dropIndexes, inplace=True)
        y_dtf.drop(dropIndexes, inplace=True)

        return X_dtf, y_dtf

    def preprocess(self, X_dtf):
        # Remplace color par la catégorie binaire color_white
        dummy = pd.get_dummies(X_dtf["color"], prefix="color", drop_first=False)
        X_dtf = pd.concat([X_dtf, dummy], axis=1)
        X_dtf = X_dtf.drop("color", axis=1)

        return X_dtf

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

        # Construit un DataFrame pour facilement manipuler les données
        df = pd.DataFrame(X_train, columns=self.column_names)
        df["id"] = df["id"].astype(int)
        df[self.convert_data] = df[self.convert_data].astype(float)
        df.set_index("id")

        y_train = pd.DataFrame(y_train, columns=["id", "quality"])
        y_train.set_index('id')

        # Pre-processing
        dtf, y_train = self.remove_outliers(df, y_train)
        dtf = self.preprocess(dtf)

        X_train = dtf[self.model_features].values
        y_train = y_train["quality"].values

        self.predictor = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=300, criterion="entropy", bootstrap=True, n_jobs=-1, random_state=0))
        #self.predictor = make_pipeline(StandardScaler(), svm.SVC(C=1.3, kernel="rbf", gamma=0.65))
        self.predictor.fit(X_train, y_train)

        #raise NotImplementedError()

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

        # Construit un DataFrame pour facilement manipuler les données
        df = pd.DataFrame(X_data, columns=self.column_names)
        df["id"] = df["id"].astype(int)
        df[self.convert_data] = df[self.convert_data].astype(float)
        df.set_index("id")

        # Pre-processing
        dtf = self.preprocess(df)

        dtf = dtf[self.model_features].values

        predictions = self.predictor.predict(dtf)

        # Mise en forme des résultats
        results = []
        for id, quality in enumerate(predictions):
            results.append([id, quality])
        return results
        #raise NotImplementedError()
