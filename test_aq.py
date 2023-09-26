import pandas as pd
from aequitas.group import Group
from aequitas.preprocessing import preprocess_input_df
from glob import glob

# IMPORTANT: Run harness.py first to generate csv test results
DIR_PATH = r'./*.csv'
test_results = glob(DIR_PATH)

for test_result in test_results:
    print(test_result)
    df = pd.read_csv(test_result)

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

    # TODO: save results to csv in a separate file
    # TODO: integrate this into the harness
    print(absolute_metrics_per_population_group)
