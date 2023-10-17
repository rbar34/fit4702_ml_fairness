from pathlib import Path
import pandas as pd

PREDICTION_DIRECTORY = r'predictions/'
GENERATED_TESTS_DIRECTORY = r'generated_tests/'

file_base_names = []
model_file = Path(PREDICTION_DIRECTORY)
for e in model_file.iterdir():
    file_base_names.append(e.name)

for file_base_name in file_base_names:
    prediction_file = Path(
        f'{PREDICTION_DIRECTORY}{file_base_name}')
    generated_tests_file = Path(
        f'{GENERATED_TESTS_DIRECTORY}{file_base_name}')

    print(f'\n{model_file}\n')

    ps = pd.read_csv(prediction_file)
    gs = pd.read_csv(generated_tests_file)

    print(ps['Predicted_Labels'].describe())
    print(gs['Predicted_Labels'].describe())
