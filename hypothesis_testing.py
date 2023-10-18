from scipy.stats import spearmanr, wilcoxon
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

TESTS_RUN = 1000

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

ALL_METRICS = GROUP_METRICS + ['ftu']

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
    for group_directory, individual_directory in zip(SOURCES['group'].values(), SOURCES['individual'].values()):
        # get the files
        files = filter(
            lambda filename: dataset in filename, file_base_names)
        min_max_differences = []
        ratio_failed_cases = []
        for file in files:
            dataframe = pd.read_csv(Path(f"{group_directory}{file}"))
            # select the sensitive attributes
            raw_data = dataframe.loc[dataframe['attribute_name']
                                     == sensitive_attribute, GROUP_METRICS]
            min_max_differences.append(raw_data.max() - raw_data.min())

            # add individual fairness
            # get the number of lines
            with open(f'{individual_directory}{file}', "r") as f:
                failed_cases = sum(1 for _ in f)
            f.close()
            ratio_failed_cases.append((failed_cases - 1)/TESTS_RUN)
        aggregated_metrics = pd.DataFrame(min_max_differences)
        aggregated_metrics["ftu"] = ratio_failed_cases
        results.append(aggregated_metrics)

    # Wilcoxon test
    # for each metric: random vs directed
    for metric in ALL_METRICS:
        score = wilcoxon(results[0][metric], results[1][metric])
        wilcoxon_df.loc[metric, "Wilcoxon"] = score.statistic
        wilcoxon_df.loc[metric, "p_value"] = score.pvalue

    # ensure directory exists
    filepath = Path(f'./statistical_results/rq1_{dataset}.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # write to file
    wilcoxon_df.to_csv(filepath, index=True)


print('\nRQ2')
print('-'*10)
# RQ2: Spearman rho testing
for dataset, sensitive_attribute in SENSITIVE_ATTRIBUTES.items():
    correlations_df = pd.DataFrame(columns=ALL_METRICS, index=ALL_METRICS)
    p_values_df = pd.DataFrame(columns=ALL_METRICS, index=ALL_METRICS)
    files = filter(lambda filename: dataset in filename, file_base_names)
    min_max_differences = []
    ratio_failed_cases = []
    print(f'\n{dataset}')
    for file in files:
        for group_directory, individual_directory in zip(SOURCES['group'].values(), SOURCES['individual'].values()):
            dataframe = pd.read_csv(Path(f"{group_directory}{file}"))
            # select the sensitive attributes
            raw_data = dataframe.loc[dataframe['attribute_name']
                                     == sensitive_attribute, GROUP_METRICS]
            min_max_differences.append(raw_data.max() - raw_data.min())

            # add individual fairness
            # get the number of lines
            with open(f'{individual_directory}{file}', "r") as f:
                failed_cases = sum(1 for _ in f)
            f.close()
            ratio_failed_cases.append((failed_cases - 1)/TESTS_RUN)
    aggregated_metrics = pd.DataFrame(min_max_differences)
    aggregated_metrics["ftu"] = ratio_failed_cases
    # Add fairness through unawareness
    # Spearman rho test
    # for each metric vs each other metric
    for metric_a in ALL_METRICS:
        for metric_b in ALL_METRICS:
            if metric_a != metric_b:
                score = spearmanr(
                    aggregated_metrics[metric_a], aggregated_metrics[metric_b])
                correlations_df.loc[metric_a, metric_b] = score.statistic
                p_values_df.loc[metric_a, metric_b] = score.pvalue

    # create pair-plot
    pairplot = sns.pairplot(data=aggregated_metrics)
    figure_filepath = Path(f'./statistical_results/rq2_{dataset}_pairplot.png')
    figure_filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fname=figure_filepath, format='png')

    # save correlations and p-values
    correltions_filepath = Path(
        f'./statistical_results/rq2_{dataset}_correlation.csv')
    pvalues_filepath = Path(f'./statistical_results/rq2_{dataset}_pvalues.csv')
    correltions_filepath.parent.mkdir(parents=True, exist_ok=True)
    pvalues_filepath.parent.mkdir(parents=True, exist_ok=True)

    correlations_df.to_csv(correltions_filepath, index=True, na_rep='NA')
    p_values_df.to_csv(pvalues_filepath, index=True, na_rep='NA')
