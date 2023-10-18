from scipy.stats import spearmanr, wilcoxon
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
    'tpr', 'fpr', 'tnr', 'fnr', 'ppr', 'precision',
]

SENSITIVE_ATTRIBUTES = {
    'german': 'personal_status_sex_',
    'adult': 'sex',
    'compas': 'sex',
}

file_base_names = []
for e in Path(DIRECTORIES['group']['random_group']).iterdir():
    file_base_names.append(e.name)

# RQ1: KS testing
print('\nRQ1')
print('-'*10)
for dataset, sensitive_attribute in SENSITIVE_ATTRIBUTES.items():
    results = []
    for test_approach, directory in DIRECTORIES['group'].items():
        print('\n', test_approach, dataset)
        # get the files
        group_files = filter(lambda filename: dataset in filename, file_base_names)
        min_max_differences = []
        for group_file in group_files:
            dataframe = pd.read_csv(Path(f"{directory}{group_file}"))
            # select the sensitive attributes
            raw_data = dataframe.loc[dataframe['attribute_name'] == sensitive_attribute, GROUP_METRICS]
            min_max_differences.append(raw_data.max() - raw_data.min())
        aggregated_metrics = pd.DataFrame(min_max_differences)
        results.append(aggregated_metrics)
        # KS test
        # for each metric: random vs directed
        # print(aggregated_metrics)
    for group_metric in GROUP_METRICS:
        print(wilcoxon(results[0][group_metric], results[1][group_metric]))


print('\nRQ2')
print('-'*10)
# RQ2: Spearman rho testing
for dataset, sensitive_attribute in SENSITIVE_ATTRIBUTES.items():
    group_files = filter(lambda filename: dataset in filename, file_base_names)
    min_max_differences = []
    print('\n', dataset)
    for group_file in group_files:
        for test_approach, directory in DIRECTORIES['group'].items():
            dataframe = pd.read_csv(Path(f"{directory}{group_file}"))
            # select the sensitive attributes
            raw_data = dataframe.loc[dataframe['attribute_name'] == sensitive_attribute, GROUP_METRICS]
            min_max_differences.append(raw_data.max() - raw_data.min())
    aggregated_metrics = pd.DataFrame(min_max_differences)
    # Spearman rho test
    # for each metric: random vs directed
    # print(aggregated_metrics)
    for metric_a in GROUP_METRICS:
        for metric_b in GROUP_METRICS:
            if metric_a != metric_b:
                print(spearmanr(aggregated_metrics[metric_a], aggregated_metrics[metric_b]))

# calculate the individual test metrics from a) the naive random, b) the directed approach

# RQ1
# perform repeated measure t-tests grouped by each metric, comparing the means of a) the naive random, b) the directed approach

# RQ2
# perform spearman rho of each metric against each other metric
