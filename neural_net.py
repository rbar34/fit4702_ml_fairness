import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score
import dataset


class NeuralNetPipeline:
    def __init__(self, data_url, model=None, test_size=0.2, random_state=42):
        self.data_url = data_url
        self.data = None
        self.X = None
        self.y = None
        self.model = model or MLPClassifier(
            hidden_layer_sizes=(5, 2), max_iter=1000)
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred_train = None
        self.y_pred_test = None

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

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def make_predictions(self):
        self.y_pred_train = self.model.predict(self.X_train)
        self.y_pred_test = self.model.predict(self.X_test)

    def evaluate_model(self):
        train_accuracy = accuracy_score(self.y_train, self.y_pred_train)
        test_accuracy = accuracy_score(self.y_test, self.y_pred_test)
        test_recall = recall_score(self.y_test, self.y_pred_test)
        return train_accuracy, test_accuracy, test_recall

    def run_pipeline(self):
        self.load_and_preprocess_data()
        self.split_data()
        self.train_model()
        self.make_predictions()
        train_acc, test_acc, test_recall = self.evaluate_model()
        print(f"Training Accuracy: {train_acc*100:.2f}%")
        print(f"Test Accuracy: {test_acc*100:.2f}%")
        print(f"Test Recall: {test_recall*100:.2f}%")


# German credit (1 (good credit) 0 (bad credit))
pipeline = NeuralNetPipeline(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")
pipeline.run_pipeline()


# Crime( 1 (High crime) 0 (low crime))
pipeline = NeuralNetPipeline(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data")
pipeline.run_pipeline()


# Adult Income (1 (> $50K) 0 (< $50K)))
pipeline = NeuralNetPipeline(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data")
pipeline.run_pipeline()
