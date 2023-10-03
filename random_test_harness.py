from random_script import RandomTestMethod
from pathlib import Path

MODEL_DIRECTORY = r'models/'
PREDICTION_DIRECTORY = r'predictions/'

file_base_names = []
model_file = Path(MODEL_DIRECTORY)
for e in model_file.iterdir():
    file_base_names.append(e.name)

for file_base_name in file_base_names:
    prediction_file = Path(
        f'{PREDICTION_DIRECTORY}{file_base_name}.csv')
    model_file = Path(f'{MODEL_DIRECTORY}{file_base_name}')

    sensitive_attribute = None

    if 'german' in model_file.name:
        sensitive_attribute = 'personal_status_sex_A92'
    elif 'adult' in model_file.name:
        sensitive_attribute = 'sex_Male'

    print(f'\n{model_file}\n')

    random_test = RandomTestMethod(
        predictions_file_path=prediction_file,
        model_file_path=model_file,
        sensitive_attribute=sensitive_attribute,
        target_variable='Predicted_Labels')

    random_test.run_random_tests()
