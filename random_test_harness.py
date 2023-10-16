from random_script import RandomTestMethod
from pathlib import Path

MODEL_DIRECTORY = r'models/'
PREDICTION_DIRECTORY = r'predictions/'
GENERATED_TESTS_DIRECTORY = r'generated_tests/'
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
            f'{GENERATED_TESTS_DIRECTORY}{file_base_name}_{i}.csv')

        sensitive_attribute = None
        categoical_attributes = None

        if 'german' in model_file.name:
            sensitive_attribute = 'sex'
        elif 'adult' in model_file.name:
            sensitive_attribute = 'sex'
        elif 'compas' in model_file.name:
            sensitive_attribute = 'sex'

        print(f'\nmodel file: {model_file}, repeat:{i}\n')

        random_test = RandomTestMethod(
            predictions_file_path=prediction_file,
            model_file_path=model_file,
            sensitive_attribute=sensitive_attribute,
            target_variable='Predicted_Labels')

        random_test.run_random_tests()
        random_test.save_failed_cases_to_csv(generated_tests_file)
