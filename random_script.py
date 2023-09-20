import numpy as np
import pandas as pd


def random_test(data_frame, sensitive_attribute, target_variable, model, metric, threshold):
    tests_ran, tests_failed = 0, 0
    while (tests_ran < threshold):
        individual_a, individual_b = generate_individuals(
            data_frame, sensitive_attribute)

        if (not metric(individual_a, individual_b, target_variable)):
            tests_failed += 1
        tests_ran += 1
    return tests_failed


def generate_individuals(data_frame, sensitive_attribute):
    # get a random invidual from the data
    individual_a = data_frame.iloc[np.random.randint(data_frame.shape[0])]
    individual_b = individual_a.copy()

    # get the categories for the sensitive attribute
    categories = list(data_frame[sensitive_attribute].cat.categories)
    categories.remove(individual_a[sensitive_attribute])

    # choose another value
    changed_sensitive_value = np.random.choice(categories)

    # TODO: need to re-run the model to re-classify
    individual_b[sensitive_attribute] = changed_sensitive_value
    return individual_a, individual_b


def fairness_through_awareness(individual_a, individual_b, target_variable):
    return individual_a[target_variable] == individual_b[target_variable]


if __name__ == '__main__':
    df = pd.read_csv("compas.csv")

    # convert non-numeric columns to categories
    df['race'] = df['race'].astype("category")

    tests_failed = random_test(df, 'race', 'two_year_recid', None,
                               fairness_through_awareness, 10)

    print(tests_failed)
