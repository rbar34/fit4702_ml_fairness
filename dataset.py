import pandas as pd
import numpy as np
from urllib.request import urlopen
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder

class Dataset:
    def __init__(self, url):
        self.url = url
        self.data = self.load_data()
        self.original_data = self.data.copy()
        self.X = None
        self.y = None

    def load_data(self):
        # Load the dataset from the URL
        data = pd.read_csv(self.url, header=None, delim_whitespace=True)
        return data

    def preprocess(self):
        if 'german.data' in self.url:  # Updated the URL snippet check
            self.preprocess_german_credit()
        elif 'adult.data' in self.url:  # Placeholder for the adult dataset URL
            self.preprocess_adult()
        elif 'communities.data' in self.url:  # Placeholder for the communities and crime dataset URL
            self.preprocess_communities_and_crime()
        else:
            raise ValueError("BAD URL")

    def preprocess_german_credit(self):
        # Add column names
        column_names = [
            "checking_status", "duration", "credit_history", "purpose", "credit_amount",
            "savings_status", "employment_duration", "installment_rate", "personal_status_sex",
            "other_debtors", "present_residence", "property", "age", "other_installment_plans",
            "housing", "existing_credits", "job", "num_liable", "phone", "foreign_worker", "creditability"
        ]
        self.data.columns = column_names

        # Map creditability to binary
        self.data["creditability"] = self.data["creditability"].map({1: 1, 2: 0})

        # Convert categorical data to dummies
        self.data = pd.get_dummies(self.data, drop_first=True)

        # Assign target & feature variables
        self.X = self.data.drop("creditability", axis=1)
        self.y = self.data["creditability"]

        # Remove statistically insignificant columns from df
        self.statistic_significance_test()
        self.X = self.data

    def preprocess_adult(self):
        # Rename the columns
        column_names = [
            "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country", "income"
        ]
        self.data.columns = column_names

        # Replace '?' with NaN and then drop rows with NaN values.
        self.data.replace(' ?', np.nan, inplace=True)
        self.data.dropna(inplace=True)

        # Check unique values in 'income' for any unexpected values
        self.data['income'] = self.data['income'].map({'<=50K': 1, '>50K': 0})

        # Encode categorical variables using one-hot encoding
        categorical_cols = [
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"
        ]

        self.data = pd.get_dummies(self.data, columns=categorical_cols, drop_first=True)

        # Assign target & feature variables
        self.X = self.data.drop("income", axis=1)
        self.y = self.data["income"]

        # Remove statistically insignificant columns from df
        self.statistic_significance_test()

        self.X = self.data

    def preprocess_communities_and_crime(self):
        # Read data into dataframe
        self.data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data",
                                header=None, na_values=["?"])
        # Read column names
        with urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.names") as names:
            columns = [line.split(b' ')[1].decode("utf-8") for line in names if line.startswith(b'@attribute')]
        self.data.columns = columns

        # Drop columns with NaN values
        self.data = self.data.dropna(axis=1)

        # Exclude first 3 columns
        self.data = self.data.iloc[:, 3:]

        # Drop non-predictive fields
        non_predictive = ['state', 'county', 'community', 'communityname', 'fold']
        self.data.drop(columns=non_predictive, inplace=True, errors='ignore')

        # Thresholding based on 70th percentile
        threshold_value = self.data['ViolentCrimesPerPop'].quantile(0.7)
        self.data['ViolentCrimesPerPop'] = self.data['ViolentCrimesPerPop'].apply(
            lambda x: 0 if x > threshold_value else 1)

        # Set target & features
        self.y = self.data['ViolentCrimesPerPop']
        self.X = self.data.drop(columns='ViolentCrimesPerPop')

        # Remove statistically insignificant columns from df
        self.statistic_significance_test()

        self.X = self.data

    def statistic_significance_test(self):
        # Perform a chi-squared test to identify statistically significant features
        chi2_scores = chi2(self.X, self.y)
        p_values = chi2_scores[1]

        # Drop features with p-values greater than 0.05
        significant_features = self.X.columns[p_values < 0.05]

        self.data = self.data[significant_features]