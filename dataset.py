import pandas as pd
from enum import Enum
import numpy as np
from urllib.request import urlopen


class Std(Enum):
    STANDARDIZED = 1
    NON_STANDARDIZED = 2


class Dataset:
    def __init__(self, url):
        self.url = url
        self.data = self.load_data()
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
            raise ValueError("fked URL")

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
        self.data["creditability"] = self.data["creditability"].map({
                                                                    1: 0, 2: 1})

        # Convert categorical data to dummies
        self.data = pd.get_dummies(self.data, drop_first=True)

        # Assign target & feature variables
        self.X = self.data.drop("creditability", axis=1)
        self.y = self.data["creditability"]

    def preprocess_adult(self):
        # Add column names
        column_names = [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "income",
        ]
        self.data.columns = column_names

        # rename columns
        self.data = self.data.rename(columns=dict([
            (str(i), e) for i, e in enumerate(column_names)
            ]))

        # Drop columns with NaN values
        self.data = self.data.dropna(axis=1)

        # Convert categorical data to dummies
        self.data = pd.get_dummies(self.data, drop_first=True)

        # Fill missing values with the column mean
        self.data.fillna(self.data.mean(), inplace=True)

        # Set target & features
        self.y = self.data["income_>50K"]
        self.X = self.data.drop(columns="income_>50K")

    def preprocess_communities_and_crime(self):
        # Read data into dataframe
        self.data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data",
                                header=None, na_values=["?"])
        # Read column names
        with urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.names") as names:
            columns = [line.split(b' ')[1].decode("utf-8")
                       for line in names if line.startswith(b'@attribute')]
        self.data.columns = columns

        # Drop columns with NaN values
        self.data = self.data.dropna(axis=1)

        # Exclude first 3 columns (you had `iloc[:, 3:]`, which means you're excluding them)
        self.data = self.data.iloc[:, 3:]

        # Drop non-predictive fields
        non_predictive = ['state', 'county',
                          'community', 'communityname', 'fold']
        self.data.drop(columns=non_predictive, inplace=True, errors='ignore')

        # Fill missing values with the column mean
        self.data.fillna(self.data.mean(), inplace=True)

        # Thresholding based on 70th percentile
        threshold_value = self.data['ViolentCrimesPerPop'].quantile(0.7)
        self.data['ViolentCrimesPerPop'] = self.data['ViolentCrimesPerPop'].apply(
            lambda x: 1 if x > threshold_value else 0)

        # Set target & features
        self.y = self.data['ViolentCrimesPerPop']
        self.X = self.data.drop(columns='ViolentCrimesPerPop')


if __name__ == '__main__':
    d = Dataset(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data")
    print(d.data)
