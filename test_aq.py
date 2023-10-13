import pandas as pd
from aequitas.group import Group
from aequitas.preprocessing import preprocess_input_df
from glob import glob
from pathlib import Path

# IMPORTANT: Run harness.py and random_test_harness.py first to generate csv test results

PATHS = {
    "aequitas": r'./predictions/*.csv',
    "random_generated_group_fairness": r'./generated_tests/*.csv'
}


for destination, predictions in PATHS.items():

    prediction_files = glob(predictions)

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
        categorisations = xtab[[col for col in xtab.columns if col not in absolute_metrics]]
        print(categorisations.to_string())

        absolute_metrics_per_population_group = xtab[[
            'attribute_name', 'attribute_value'] + absolute_metrics].round(2)

        # TODO: integrate this into the harness
        pathname = f"./{destination}/{prediction_file.split('/')[-1]}"
        filepath = Path(pathname)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        absolute_metrics_per_population_group.to_csv(filepath, index=False)
