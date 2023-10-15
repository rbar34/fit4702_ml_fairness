import pandas as pd
from aequitas.group import Group
from aequitas.preprocessing import preprocess_input_df
from glob import glob
from pathlib import Path

PATHS = {
    "aequitas": r'./total_tests/*.csv',
    "random_generated_group_fairness": r'./generated_tests/*.csv'
}


for destination, predictions in PATHS.items():

    prediction_files = glob(predictions)

    for prediction_file in prediction_files:
        print(prediction_file)

        if 'german' in prediction_file:
            sensitive_attribute = 'personal_status_sex_'
            separator = 'A'
            default_category = {sensitive_attribute: '99'}
        elif 'adult' in prediction_file:
            sensitive_attribute = 'sex'
            separator = '_'
            default_category = {sensitive_attribute: 'Female'}
        elif 'compas' in prediction_file:
            sensitive_attribute = 'sex'
            separator = '_'
            default_category = {sensitive_attribute: 'Female'}

        df = pd.read_csv(prediction_file)

        sensitive_attribute_indexes = df.columns.to_series().str.contains(sensitive_attribute)

        dummy_columns = pd.DataFrame(df.loc[:, sensitive_attribute_indexes])

        undummied = pd.from_dummies(
            data=dummy_columns, sep=separator, default_category=default_category)

        # remove dummy columns and replace with undummied column
        df.drop(columns=df.columns[sensitive_attribute_indexes], inplace=True)
        df.loc[:, sensitive_attribute] = undummied

        # rename columns to work properly with AEQUITAS
        df.rename(columns={'Predicted_Labels': 'score',
                  'True_Labels': 'label_value'}, inplace=True)

        # preprocess for AEQUITAS
        df, _ = preprocess_input_df(df)

        # compute group metrics
        g = Group()
        xtab, _ = g.get_crosstabs(df)
        absolute_metrics = g.list_absolute_metrics(xtab)
        categorisations = xtab[[
            col for col in xtab.columns if col not in absolute_metrics]]
        print(categorisations.to_string())

        absolute_metrics_per_population_group = xtab[[
            'attribute_name', 'attribute_value'] + absolute_metrics].round(2)

        # TODO: integrate this into the harness
        pathname = f"./{destination}/{prediction_file.split('/')[-1]}"
        filepath = Path(pathname)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        absolute_metrics_per_population_group.to_csv(filepath, index=False)
