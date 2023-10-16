from scipy.stats import spearmanr, ttest_rel
import pandas as pd
from glob import glob

DIRECTORIES = {
    "random_individual": r'./generated_tests',
    "random_group": r'./random_generated_group_fairness/',
    "directed_individual": r'./total_tests/',
    "directed_group": r'./aequitas/',
}

random_group_files = glob(DIRECTORIES['random_group'])
for random_group_file in random_group_files:
    if 'german' in random_group_file:
        sensitive_attribute = 'personal_status_sex_'
        separator = 'A'
        default_category = {sensitive_attribute: '99'}
    elif 'adult' in random_group_file:
        sensitive_attribute = 'sex'
        separator = '_'
        default_category = {sensitive_attribute: 'Female'}
    elif 'compas' in random_group_file:
        sensitive_attribute = 'sex'
        separator = '_'
        default_category = {sensitive_attribute: 'Female'}

# separate by datasets


# LOADING DATA
# import the aequitas group metrics from a) the naive random, b) the directed approach


# calculate the individual test metrics from a) the naive random, b) the directed approach

# CALCULATING
# group metrics: calculate the max-min metrics from each file in a) the naive random, b) the directed approach

# RQ1
# perform repeated measure t-tests grouped by each metric, comparing the means of a) the naive random, b) the directed approach

# RQ2
# perform spearman rho of each metric against each other metric
