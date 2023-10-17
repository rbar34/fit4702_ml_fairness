import numpy as np
import pandas as pd
import time
from joblib import load
from pathlib import Path
from scipy.optimize import basinhopping

MODEL_DIRECTORY = r'models/'
PREDICTION_DIRECTORY = r'predictions/'
TESTS_TO_RUN = 1000


class DirectedTestMethod:

    def __init__(self, predictions_file_path, model_file_path, sensitive_attribute, target_variable, random_state=42):
        self.predictions_file_path = predictions_file_path
        self.model_file_path = model_file_path
        self.predictions = None
        self.model = None
        self.sensitive_attribute = sensitive_attribute
        self.target_variable = target_variable
        self.sensitive_param_index = -1
        self.param_probability_change_size = 0.001
        self.direction_probability_change_size = 0.001
        self.init_prob = 0.5
        self.threshold = 1
        self.stepsize = 1
        self.perturbation_unit = 1
        self.global_iteration_limit = 1000
        self.local_iteration_limit = 1000
        np.random.seed(random_state)

    def load_prediction_model_pair(self):
        self.model = load(self.model_file_path)
        self.predictions = pd.read_csv(self.predictions_file_path)

    def save_failed_cases_to_csv(self, filename):
        failed_tests = self.global_disc_inputs_list + self.local_disc_inputs_list
        result_df = pd.DataFrame(failed_tests, columns=self.predictions.columns.values[:-3])
        result_df[self.dataframe_bool_types.columns] = result_df[self.dataframe_bool_types.columns].astype(bool)

        #result_df.columns = self.predictions.columns.values
        # ensure directory exists
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # write to file
        result_df.to_csv(filepath, index=False)

    def df_min_max(self, df):
        return pd.Series(index=['min','max'],data=[df.min(),df.max()])
    
    def directed_tests(self, data_frame, sensitive_attribute, target_variable):
        data_frame = data_frame.iloc[:, :-3]
        self.dataframe_bool_types = data_frame.select_dtypes(bool)
        data_frame[self.dataframe_bool_types.columns] = self.dataframe_bool_types.astype(int)

        self.sensitive_param_index = list(data_frame.columns.values).index(sensitive_attribute)

        self.param_count = data_frame.shape[1]
        self.param_probability = [1.0/self.param_count] * self.param_count
        self.direction_probability = [self.init_prob] * self.param_count
        self.input_bounds = data_frame.apply(self.df_min_max).T.values.tolist()
        
        self.global_disc_inputs = set()
        self.global_disc_inputs_list = []

        self.local_disc_inputs = set()
        self.local_disc_inputs_list = []

        self.tot_inputs = set()

        initial_input = data_frame.iloc[np.random.randint(
                 data_frame.shape[0])]
        
        minimizer = {"method": "L-BFGS-B"}

        basinhopping(self.evaluate_global, initial_input, stepsize=1.0, take_step=self.global_discovery,
             minimizer_kwargs=minimizer, niter=self.global_iteration_limit)
        
        print("Finished Global Search")
        print("Percentage discriminatory inputs - " + str(float(len(self.global_disc_inputs_list)
                                                                + len(self.local_disc_inputs_list)) / float(len(self.tot_inputs))*100))
        print("")
        print("Starting Local Search")

        for inp in self.global_disc_inputs_list:
                basinhopping(self.evaluate_local, inp, stepsize=1.0, take_step=self.local_perturbation, minimizer_kwargs=minimizer,
                 niter=self.local_iteration_limit)

        print("")
        print("Local Search Finished")
        print("Percentage discriminatory inputs - " + str(float(len(self.global_disc_inputs_list) + len(self.local_disc_inputs_list))
                                                        / float(len(self.tot_inputs))*100))

        print("")
        print("Total Inputs are " + str(len(self.tot_inputs)))
        print("Number of discriminatory inputs are " + str(len(self.global_disc_inputs_list)+len(self.local_disc_inputs_list)))

        return len(self.global_disc_inputs_list + self.local_disc_inputs_list)

    def permute_dummy_encoded_attribute(self, individual_a, attribute):
        individual_b = individual_a.copy()

        attribute_indices = individual_b.index.to_series().str.contains(attribute)
        dummy_columns = individual_b.index[attribute_indices]
        permuted_columns = [0] * len(dummy_columns)
        random_index = np.random.randint(len(dummy_columns) + 1)
        if random_index < len(dummy_columns):
            permuted_columns[random_index] = 1
        individual_b[dummy_columns] = permuted_columns

        return individual_b

    def fairness_through_awareness(individual_a, individual_b, target_variable):
        return individual_a[target_variable] == individual_b[target_variable]

    def run_directed_tests(self):
        self.load_prediction_model_pair()
        results = self.directed_tests(data_frame=self.predictions,
                                   sensitive_attribute=self.sensitive_attribute,
                                   target_variable=self.target_variable)
        print(f'failed cases out of {TESTS_TO_RUN}: {results}')

    def evaluate_input(self, inp):
        inp0 = [int(i) for i in inp]
        inp1 = [int(i) for i in inp]

        inp0[self.sensitive_param_index - 1] = 0
        inp1[self.sensitive_param_index - 1] = 1

        inp0 = np.asarray(inp0)
        inp0 = np.reshape(inp0, (1, -1))

        inp1 = np.asarray(inp1)
        inp1 = np.reshape(inp1, (1, -1))

        out0 = self.model.predict_wrapper(inp0)
        out1 = self.model.predict_wrapper(inp1)
        #return abs(out0 - out1) > threshold
        #for binary classification, we have found that the
        #following optimization function gives better results
        return abs(out1 + out0) == 0

    def evaluate_local(self, inp):
        inp0 = [int(i) for i in inp]
        inp1 = [int(i) for i in inp]

        inp0[self.sensitive_param_index - 1] = 0
        inp1[self.sensitive_param_index - 1] = 1

        inp0 = np.asarray(inp0)
        inp0 = np.reshape(inp0, (1, -1))

        inp1 = np.asarray(inp1)
        inp1 = np.reshape(inp1, (1, -1))

        out0 = self.model.predict_wrapper(inp0)
        out1 = self.model.predict_wrapper(inp1)

        self.tot_inputs.add(tuple(map(tuple, inp0)))

        if (abs(out0 - out1) != 0 and (tuple(map(tuple, inp0)) not in self.global_disc_inputs)
            and (tuple(map(tuple, inp0)) not in self.local_disc_inputs)):
            self.local_disc_inputs.add(tuple(map(tuple, inp0)))
            self.local_disc_inputs_list.append(inp0.tolist()[0])

        return abs(out0 + out1)
    
    def evaluate_global(self, inp):
        inp0 = [int(i) for i in inp]
        inp1 = [int(i) for i in inp]

        inp0[self.sensitive_param_index] = 0
        inp1[self.sensitive_param_index] = 1

        inp0 = np.asarray(inp0)
        inp0 = np.reshape(inp0, (1, -1))

        inp1 = np.asarray(inp1)
        inp1 = np.reshape(inp1, (1, -1))

        out0 = self.model.predict_wrapper(inp0)
        out1 = self.model.predict_wrapper(inp1)

        self.tot_inputs.add(tuple(map(tuple, inp0)))

        if (abs(out0 - out1) != 0 and tuple(map(tuple, inp0)) not in self.global_disc_inputs):
            self.global_disc_inputs.add(tuple(map(tuple, inp0)))
            self.global_disc_inputs_list.append(inp0.tolist()[0])

        # return not abs(out0 - out1) > threshold
        #for binary classification, we have found that the
        #following optimization function gives better results
        return abs(out1 + out0)

    def local_perturbation(self, x):
        param_choice = np.random.choice(range(self.param_count) , p=self.param_probability)
        perturbation_options = [-1, 1]

        # choice = np.random.choice(perturbation_options)
        direction_choice = np.random.choice(perturbation_options, p=[self.direction_probability[param_choice],
                                                                     (1 - self.direction_probability[param_choice])])

        if (x[param_choice] == self.input_bounds[param_choice][0]) or (x[param_choice] == self.input_bounds[param_choice][1]):
            direction_choice = np.random.choice(perturbation_options)

        x[param_choice] = x[param_choice] + (direction_choice * self.perturbation_unit)

        x[param_choice] = max(self.input_bounds[param_choice][0], x[param_choice])
        x[param_choice] = min(self.input_bounds[param_choice][1], x[param_choice])

        ei = self.evaluate_input(x)

        if (ei and direction_choice == -1) or (not ei and direction_choice == 1):
            self.direction_probability[param_choice] = min(self.direction_probability[param_choice] +
                                                      (self.direction_probability_change_size * self.perturbation_unit), 1)

        elif (not ei and direction_choice == -1) or (ei and direction_choice == 1):
            self.direction_probability[param_choice] = max(self.direction_probability[param_choice] -
                                                      (self.direction_probability_change_size * self.perturbation_unit), 0)

        if ei:
            self.param_probability[param_choice] = self.param_probability[param_choice] + self.param_probability_change_size
            self.normalise_probability()
        else:
            self.param_probability[param_choice] = max(self.param_probability[param_choice] - self.param_probability_change_size, 0)
            self.normalise_probability()

        return x
    
    def global_discovery(self, x):
        for i in range(self.param_count):
            if self.input_bounds[i][0] == self.input_bounds[i][1]:
                 x[i] =  self.input_bounds[i][0]
                 continue
            
            x[i] = np.random.randint(self.input_bounds[i][0], self.input_bounds[i][1])

        x[self.sensitive_param_index] = 0
        return x

    def normalise_probability(self):
        probability_sum = 0.0
        for prob in self.param_probability:
            probability_sum = probability_sum + prob

        for i in range(len(self.param_probability)):
            self.param_probability[i] = float(self.param_probability[i])/float(probability_sum)