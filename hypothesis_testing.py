from scipy.stats import spearmanr, ttest_rel
import pandas as pd
from glob import glob
from pathlib import Path

DIRECTORIES = {
    "individual": {
        "random_individual": r'./failed_tests_random/',
        "directed_individual": r'./failed_tests_directed/', },
    "group": {
        "random_group": r'./group_metrics_random/',
        "directed_group": r'./group_metrics_directed/', },
}

GROUP_METRICS = [
    'tpr', 'fpr', 'tnr', 'fnr', 'precision',
]

SENSITIVE_ATTRIBUTES = {
    'german': 'personal_status_sex_',
    'adult': 'sex',
    'compas': 'sex',
}

file_base_names = []
for e in Path(DIRECTORIES['group']['random_group']).iterdir():
    file_base_names.append(e.name)

# RQ1: KS Testing
for dataset, sensitive_attribute in SENSITIVE_ATTRIBUTES.items():
    for test_approach, directory in DIRECTORIES['group'].items():
        print(test_approach, dataset)
        # get the files
        group_files = filter(lambda filename: dataset in filename, file_base_names)
        min_max_differences = []
        for group_file in group_files:
            dataframe = pd.read_csv(Path(f"{directory}{group_file}"))
            # select the sensitive attributes
            raw_data = dataframe.loc[dataframe['attribute_name'] == sensitive_attribute, GROUP_METRICS]
            min_max_differences.append(raw_data.max() - raw_data.min())
        aggregated_metrics = pd.DataFrame(min_max_differences)
        # KS Test
        # for each metric: random vs directed
        print(aggregated_metrics)

# calculate the individual test metrics from a) the naive random, b) the directed approach

# RQ1
# perform repeated measure t-tests grouped by each metric, comparing the means of a) the naive random, b) the directed approach

# RQ2
# perform spearman rho of each metric against each other metric
