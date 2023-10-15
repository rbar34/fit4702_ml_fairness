import numpy as np
import pandas as pd
from joblib import load
from pathlib import Path

MODEL_DIRECTORY = r'models/'
PREDICTION_DIRECTORY = r'predictions/'
TESTS_TO_RUN = 1000


class TotalTestMethod:

    def __init__(self, predictions_file_path, model_file_path, sensitive_attribute, predictive_attributes, target_variable):
        self.predictions_file_path = predictions_file_path
        self.model_file_path = model_file_path
        self.predictions = None
        self.model = None
        self.sensitive_attribute = sensitive_attribute
        self.target_variable = target_variable
        self.predictive_attributes = predictive_attributes  # categorical only
        self.failed_cases = {}

    def load_prediction_model_pair(self):
        self.model = load(self.model_file_path)
        self.predictions = pd.read_csv(self.predictions_file_path)

    def save_failed_cases_to_csv(self, filename):
        result_df = pd.DataFrame(self.failed_cases.values())

        # ensure directory exists
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # write to file
        result_df.to_csv(filepath, index=False)

    def total_test(self, data_frame, metric):
        tests_failed = 0

        individual_a = data_frame.iloc[np.random.randint(data_frame.shape[0])]

        for _ in range(TESTS_TO_RUN):
            individual_b = self.permute_dummy_encoded_attribute(
                individual_a, self.sensitive_attribute)

            metric_failed = self.test_individuals_against_metric(
                individual_a, individual_b, metric)

            if metric_failed:
                # record failed cases
                tests_failed += 1
                id = str(pd.util.hash_pandas_object(individual_a, index=False))
                self.failed_cases[id] = individual_a

                # produce a similar individual
                attribute = np.random.choice(self.predictive_attributes)
                individual_a = self.permute_dummy_encoded_attribute(
                    individual_a, attribute)
            else:
                # get another individual at random
                individual_a = data_frame.iloc[np.random.randint(
                    data_frame.shape[0])]
        return tests_failed

    def permute_dummy_encoded_attribute(self, individual_a, attribute):
        individual_b = individual_a.copy()

        attribute_indices = individual_b.index.to_series().str.contains(attribute)
        dummy_columns = individual_b.index[attribute_indices]
        permuted_columns = [0] * len(dummy_columns)
        permuted_columns[np.random.randint(len(dummy_columns))] = 1
        individual_b[dummy_columns] = permuted_columns

        return individual_b

    def test_individuals_against_metric(self, individual_a, individual_b, metric):
        # Remove prediction and label columns before predicting
        test_b = individual_b.iloc[:-3]
        test_b = np.asarray(test_b).astype('float32')
        test_b = np.reshape(test_b, (1, -1))
        individual_b[self.target_variable] = self.model.predict_wrapper(test_b)

        return (not metric(individual_a, individual_b, self.target_variable))

    def fairness_through_unawareness(individual_a, individual_b, target_variable):
        return individual_a[target_variable] == individual_b[target_variable]

    def run_total_tests(self):
        self.load_prediction_model_pair()
        results = self.total_test(data_frame=self.predictions,
                                  metric=TotalTestMethod.fairness_through_unawareness,
                                  )
        print(f'failed cases out of {TESTS_TO_RUN} : {results}')
