import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import dataset
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, recall_score, precision_score
from pathlib import Path
from joblib import dump


class BaseModelPipeline:
    def __init__(self, data_url, model, test_size=0.2, random_state=42,scale_data=True):
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
        self.scale_data = scale_data

    def load_and_preprocess_data(self):
        """
            Load each data set & preprocess
        """
        ds = dataset.Dataset(self.data_url)
        ds.preprocess()
        self.data = ds.data
        self.X = ds.X
        self.y = ds.y

    def split_data(self):
        """
            Split the data set into train & test sets
        """
        # Split the train and test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=self.test_size,
                                                                                random_state=self.random_state)

        self.X_test_indices = self.X_test.index  # Save original indices of X_test so it doesn't print scaled results

        # Scale the data (enabled by default)
        if self.scale_data:
            self.X_train = self.scaler.fit_transform(self.X_train.values)
            self.X_test = self.scaler.transform(self.X_test.values)

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def make_predictions(self):
        self.y_pred_train = self.model.predict(self.X_train)
        self.y_pred_test = self.model.predict(self.X_test)

    def evaluate_model(self):
        """
            Evaluate model performance
        """
        train_accuracy = accuracy_score(self.y_train, self.y_pred_train)
        test_accuracy = accuracy_score(self.y_test, self.y_pred_test)
        train_recall = recall_score(self.y_train, self.y_pred_train)
        test_recall = recall_score(self.y_test, self.y_pred_test)
        train_precision = precision_score(self.y_train, self.y_pred_train)
        test_precision = precision_score(self.y_test, self.y_pred_test)
        train_f1_score = f1_score(self.y_train, self.y_pred_train)
        test_f1_score = f1_score(self.y_test, self.y_pred_test)
        train_roc_auc = roc_auc_score(self.y_train, self.y_pred_train)
        test_roc_auc = roc_auc_score(self.y_test, self.y_pred_test)
        return {
            'Train Accuracy': train_accuracy,
            'Train Recall': train_recall,
            'Train Precision': train_precision,
            'Train F1 Score': train_f1_score,
            'Train ROC AUC': train_roc_auc,
            'Test Accuracy': test_accuracy,
            'Test Recall': test_recall,
            'Test Precision': test_precision,
            'Test F1 Score': test_f1_score,
            'Test ROC AUC': test_roc_auc
        }

    def run_pipeline(self):
        """
            Execute pipeline on preprocessed data
        """
        self.load_and_preprocess_data()
        self.split_data()
        self.train_model()
        self.make_predictions()
        metrics = self.evaluate_model()
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value * 100:.2f}%")

    def save_model(self, filename):
        """
            Serialise model and save to filesystem
        """

        model_wrapper = ModelWrapper(self.model, self.scaler)

        # ensure directory exists
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # write to file
        dump(model_wrapper, filepath)

    def save_predictions_to_csv(self, filename):
        """
            Save predictions against actual data in a CSV
        """

        # Use the stored indices to extract the corresponding rows frm the original dataset
        X_test_original_df = self.data.loc[self.X_test_indices].reset_index(
            drop=True)

        # Drop duplicates
        X_test_original_df = X_test_original_df.drop(
            columns=["target_column_name"], errors='ignore')

        # concat the original data, predictions, and true labels
        result_df = pd.concat([
            X_test_original_df,
            pd.Series(self.y_pred_test, name="Predicted_Labels"),
            pd.Series(self.y_test, name="True_Labels").reset_index(drop=True)
        ], axis=1)

        # ensure directory exists
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # write to file
        result_df.to_csv(filepath, index=False)


class RandomForestPipeline(BaseModelPipeline):
    def __init__(self, data_url, test_size=0.2, random_state=42):
        super().__init__(
            data_url,
            RandomForestClassifier(
                n_estimators=100,   # Ensures sufficient number of trees
                max_depth=6,        # Decreased from 10
                min_samples_split=7,  # Increased from 5
                min_samples_leaf=7,  # Increased from 4
                # Consider sqrt(number of features) for each split
                max_features="sqrt",
                oob_score=True,
                random_state=random_state
            ),
            test_size,
            random_state
        )


class SVMPipeline(BaseModelPipeline):
    def __init__(self, data_url, C=0.3, test_size=0.2, random_state=42):
        super().__init__(data_url, SVC(C=C), test_size, random_state)


class LogisticPipeline(BaseModelPipeline):
    def __init__(self, data_url, test_size=0.2, random_state=42):
        super().__init__(data_url, LogisticRegression(
            max_iter=1000), test_size, random_state)


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
        model.compile(optimizer='adam',
                      loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self):
        self.model.fit(self.X_train, self.y_train, epochs=5,
                       batch_size=32, validation_split=0.2)

    def make_predictions(self):
        self.y_pred_train = (self.model.predict(
            self.X_train).flatten() > 0.5).astype(int)
        self.y_pred_test = (self.model.predict(
            self.X_test).flatten() > 0.5).astype(int)

    def save_model(self, filename):

        model_wrapper = NNModelWrapper(self.model, self.scaler)

        # ensure directory exists
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # write to file
        dump(model_wrapper, filepath)


class ModelWrapper:
    def __init__(self, model, scalar):
        self.model = model
        self.scaler = scalar 

    def predict_wrapper(self, X):
        X = self.scaler.transform(X)
        return self.model.predict(X)


class NNModelWrapper(ModelWrapper):
    def predict_wrapper(self, X):
        X = self.scaler.transform(X)
        return (self.model.predict(X, verbose=0).flatten() > 0.5).astype(int)
