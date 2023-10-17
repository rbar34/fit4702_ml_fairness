from directed_aq import DirectedTestMethod
from pathlib import Path

MODEL_DIRECTORY = r'models/'
PREDICTION_DIRECTORY = r'predictions/'
GENERATED_TESTS_DIRECTORY = r'directed_tests/'
NUMBER_OF_REPEATS = 1

file_base_names = []
model_file = Path(MODEL_DIRECTORY)
for e in model_file.iterdir():
    file_base_names.append(e.name)

for i in range(NUMBER_OF_REPEATS):
    for file_base_name in file_base_names:
        prediction_file = Path(
            f'{PREDICTION_DIRECTORY}{file_base_name}.csv')
        model_file = Path(f'{MODEL_DIRECTORY}{file_base_name}')
        generated_tests_file = Path(
            f'{GENERATED_TESTS_DIRECTORY}{file_base_name}.csv')

        sensitive_attribute = None
        categoical_attributes = None

        if 'german' in model_file.name:
            sensitive_attribute = 'sex'
            continue
        elif 'adult' in model_file.name:
            sensitive_attribute = 'sex_Male'
            continue
        elif 'compas' in model_file.name:
            sensitive_attribute = 'sex_Male'

        print(f'\nmodel file: {model_file}, repeat:{i}\n')

        directed_test = DirectedTestMethod(
            predictions_file_path=prediction_file,
            model_file_path=model_file,
            sensitive_attribute=sensitive_attribute,
            target_variable='Predicted_Labels')

        directed_test.run_directed_tests()
        directed_test.save_failed_cases_to_csv(generated_tests_file)