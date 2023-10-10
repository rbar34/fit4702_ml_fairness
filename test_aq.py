import pandas as pd
from aequitas.group import Group
from aequitas.preprocessing import preprocess_input_df
from glob import glob
from pathlib import Path

# IMPORTANT: Run harness.py first to generate csv test results

# TODO: write some loop to handle both paths
PATHS = {
        
        }

# DIR_PATH = r'./predictions/*.csv'
DIR_PATH = r'./generated_tests/*.csv'
prediction_files = glob(DIR_PATH)

for prediction_file in prediction_files:
    print(prediction_file)

    df = pd.read_csv(prediction_file)

    # rename columns to work properly with AEQUITAS
    df.rename(columns={'Predicted_Labels': 'score',
              'True_Labels': 'label_value'}, inplace=True)

    # preprocess for AEQUITAS
    df, _ = preprocess_input_df(df)

    # compute group metrics
    g = Group()
    xtab, _ = g.get_crosstabs(df)
    absolute_metrics = g.list_absolute_metrics(xtab)

    absolute_metrics_per_population_group = xtab[[
        'attribute_name', 'attribute_value'] + absolute_metrics].round(2)

    # TODO: integrate this into the harness
    pathname = f"./random_generated_group_fairness/{prediction_file.split('/')[-1]}"
    # pathname = f"./aequitas/{prediction_file.split('/')[-1]}"
    filepath = Path(pathname)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    absolute_metrics_per_population_group.to_csv(filepath, index=False)
    print(absolute_metrics_per_population_group)
