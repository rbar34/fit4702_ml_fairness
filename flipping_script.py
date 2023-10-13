import numpy as np
import pandas as pd
from joblib import load
from pathlib import Path

MODEL_DIRECTORY = r'models/'
PREDICTION_DIRECTORY = r'predictions/'


class TotalTestMethod:

    def __init__(self, predictions_file_path, model_file_path, sensitive_attribute, target_variable):
        self.predictions_file_path = predictions_file_path
        self.model_file_path = model_file_path
        self.predictions = None
        self.model = None
        self.sensitive_attribute = sensitive_attribute
        self.target_variable = target_variable
        self.failed_cases = []

    def load_prediction_model_pair(self):
        self.model = load(self.model_file_path)
        self.predictions = pd.read_csv(self.predictions_file_path)

    def save_failed_cases_to_csv(self, filename):
        result_df = pd.DataFrame(self.failed_cases)

        # ensure directory exists
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # write to file
        result_df.to_csv(filepath, index=False)

    def total_test(self, data_frame, sensitive_attribute, target_variable, model, metric):
        tests_failed = 0
        for i, individual_a in data_frame.iterrows():
            individual_b = individual_a.copy()
            individual_b[sensitive_attribute] = not individual_b[sensitive_attribute]

            # Remove prediction and label columns before predicting
            test_b = individual_b.iloc[:-2]
            test_b = np.asarray(test_b).astype('float32')
            test_b = np.reshape(test_b, (1, -1))
            individual_b[target_variable] = self.model.predict_wrapper(test_b)

            if (not metric(individual_a, individual_b, target_variable)):
                tests_failed += 1
                # record failed cases
                self.failed_cases += [individual_a, individual_b]
        return tests_failed

    def fairness_through_unawareness(individual_a, individual_b, target_variable):
        return individual_a[target_variable] == individual_b[target_variable]

    def run_total_tests(self):
        self.load_prediction_model_pair()
        results = self.total_test(data_frame=self.predictions,
                                  sensitive_attribute=self.sensitive_attribute,
                                  target_variable=self.target_variable,
                                  model=self.model,
                                  metric=TotalTestMethod.fairness_through_unawareness,
                                  )
        print(f'failed cases out of {len(self.predictions)} : {results}')
