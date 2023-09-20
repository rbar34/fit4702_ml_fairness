import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import dataset
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class BaseModelPipeline:
    def __init__(self, data_url, model, test_size=0.2, random_state=42):
        self.data_url = data_url
        self.data = None
        self.X = None
        self.y = None
        self.model = model
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred_train = None
        self.y_pred_test = None
        self.scaler = StandardScaler()

    def load_and_preprocess_data(self):
        ds = dataset.Dataset(self.data_url)
        ds.preprocess()
        self.data = ds.data
        self.X = ds.X
        self.y = ds.y

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=self.test_size,
                                                                                random_state=self.random_state)
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def make_predictions(self):
        self.y_pred_train = self.model.predict(self.X_train)
        self.y_pred_test = self.model.predict(self.X_test)

    def evaluate_model(self):
        train_accuracy = accuracy_score(self.y_train, self.y_pred_train)
        test_accuracy = accuracy_score(self.y_test, self.y_pred_test)
        return train_accuracy, test_accuracy

    def run_pipeline(self):
        self.load_and_preprocess_data()
        self.split_data()
        self.train_model()
        self.make_predictions()
        train_acc, test_acc = self.evaluate_model()
        print(f"Training Accuracy: {train_acc * 100:.2f}%")
        print(f"Test Accuracy: {test_acc * 100:.2f}%")

    def save_predictions_to_csv(self, filename):
        df = pd.DataFrame({
            'True_Labels': self.y_test,
            'Predicted_Labels': self.y_pred_test
        })

        df.to_csv(filename, index=False)
        print(f"Predictions saved to {filename}")


class RandomForestPipeline(BaseModelPipeline):
    def __init__(self, data_url, test_size=0.2, random_state=42):
        super().__init__(data_url, RandomForestClassifier(), test_size, random_state)


class SVMPipeline(BaseModelPipeline):
    def __init__(self, data_url, test_size=0.2, random_state=42):
        super().__init__(data_url, SVC(), test_size, random_state)


class LogisticPipeline(BaseModelPipeline):
    def __init__(self, data_url, test_size=0.2, random_state=42):
        super().__init__(data_url, LogisticRegression(max_iter=1000), test_size, random_state)


class NeuralNetworkPipeline(BaseModelPipeline):
    def __init__(self, data_url, test_size=0.2, random_state=42):
        model = self.build_default_model()
        super().__init__(data_url, model, test_size, random_state)

    def build_default_model(self):
        model = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self):
        self.model.fit(self.X_train, self.y_train, epochs=10, batch_size=32, validation_split=0.2)

    def make_predictions(self):
        self.y_pred_train = (self.model.predict(self.X_train).flatten() > 0.5).astype(int)
        self.y_pred_test = (self.model.predict(self.X_test).flatten() > 0.5).astype(int)


