from scipy.stats import spearmanr, wilcoxon
import pandas as pd
from pathlib import Path

SOURCES = {
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
for e in Path(SOURCES['group']['random_group']).iterdir():
    file_base_names.append(e.name)

# RQ1: Wilcoxon testing
print('\nRQ1')
print('-'*10)
for dataset, sensitive_attribute in SENSITIVE_ATTRIBUTES.items():
    wilcoxon_df = pd.DataFrame(
        columns=["Wilcoxon", "p_value"], index=GROUP_METRICS)
    print(f'\n{dataset}')
    results = []
    for test_approach, directory in SOURCES['group'].items():
        # get the files
        group_files = filter(
            lambda filename: dataset in filename, file_base_names)
        min_max_differences = []
        for group_file in group_files:
            dataframe = pd.read_csv(Path(f"{directory}{group_file}"))
            # select the sensitive attributes
            raw_data = dataframe.loc[dataframe['attribute_name']
                                     == sensitive_attribute, GROUP_METRICS]
            min_max_differences.append(raw_data.max() - raw_data.min())
        aggregated_metrics = pd.DataFrame(min_max_differences)
        results.append(aggregated_metrics)
        # KS test
        # for each metric: random vs directed
    for group_metric in GROUP_METRICS:
        score = wilcoxon(results[0][group_metric], results[1][group_metric])
        wilcoxon_df.loc[group_metric, "Wilcoxon"] = score.statistic
        wilcoxon_df.loc[group_metric, "p_value"] = score.pvalue

    # ensure directory exists
    filepath = Path(r'./statistical_results/rq1.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # write to file
    wilcoxon_df.to_csv(filepath, index=True)
    print(wilcoxon_df)


print('\nRQ2')
print('-'*10)
# RQ2: Spearman rho testing
for dataset, sensitive_attribute in SENSITIVE_ATTRIBUTES.items():
    correlations_df = pd.DataFrame(columns=GROUP_METRICS, index=GROUP_METRICS)
    p_values_df = pd.DataFrame(columns=GROUP_METRICS, index=GROUP_METRICS)
    group_files = filter(lambda filename: dataset in filename, file_base_names)
    min_max_differences = []
    print(f'\n{dataset}')
    for group_file in group_files:
        for test_approach, directory in SOURCES['group'].items():
            dataframe = pd.read_csv(Path(f"{directory}{group_file}"))
            # select the sensitive attributes
            raw_data = dataframe.loc[dataframe['attribute_name']
                                     == sensitive_attribute, GROUP_METRICS]
            min_max_differences.append(raw_data.max() - raw_data.min())
    aggregated_metrics = pd.DataFrame(min_max_differences)
    # Spearman rho test
    # for each metric vs each other metric
    for metric_a in GROUP_METRICS:
        for metric_b in GROUP_METRICS:
            if metric_a != metric_b:
                score = spearmanr(
                    aggregated_metrics[metric_a], aggregated_metrics[metric_b])
                correlations_df.loc[metric_a, metric_b] = score.statistic
                p_values_df.loc[metric_a, metric_b] = score.pvalue

    # ensure directory exists
    correltions_filepath = Path(f'./statistical_results/rq2_{dataset}_correlation.csv')
    pvalues_filepath = Path(f'./statistical_results/rq2_{dataset}_pvalues.csv')
    correltions_filepath.parent.mkdir(parents=True, exist_ok=True)
    pvalues_filepath.parent.mkdir(parents=True, exist_ok=True)

    # write to file
    correlations_df.to_csv(correltions_filepath, index=True, na_rep='NA')
    p_values_df.to_csv(pvalues_filepath, index=True, na_rep='NA')

    print(correlations_df)
