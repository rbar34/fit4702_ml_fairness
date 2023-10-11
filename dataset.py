import pandas as pd
import numpy as np
from urllib.request import urlopen
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

class Dataset:
    def __init__(self, url):
        self.url = url
        self.data = None
        self.X = None
        self.y = None


    def preprocess(self):
        if 'german.data' in self.url:  #
            self.preprocess_german_credit()
        elif 'adult.data' in self.url:
            self.preprocess_adult()
        elif 'compas' in self.url:
            self.preprocess_compass()
        else:
            raise ValueError("BAD URL")

    def preprocess_german_credit(self):
        # Load data
        self.data = pd.read_csv(self.url, header=None, delim_whitespace=True)
        self.data = self.data.map(lambda x: x.strip() if isinstance(x, str) else x)

        # Rename the columns
        column_names = [
            "checking_status", "duration", "credit_history", "purpose", "credit_amount",
            "savings_status", "employment_duration", "installment_rate", "personal_status_sex",
            "other_debtors", "present_residence", "property", "age", "other_installment_plans",
            "housing", "existing_credits", "job", "num_liable", "phone", "foreign_worker", "creditability"
        ]
        self.data.columns = column_names

        # Check unique values in 'creditability' for any unexpected values
        self.data["creditability"] = self.data["creditability"].map({1: 1, 2: 0})

        # Encode categorical variables using one-hot encoding
        categorical_cols = [
            "checking_status", "credit_history", "purpose", "savings_status",
            "employment_duration", "personal_status_sex", "other_debtors",
            "property", "other_installment_plans", "housing", "job", "phone", "foreign_worker"
        ]
        self.data = pd.get_dummies(self.data, columns=categorical_cols, drop_first=True)

        # Assign target & feature variables
        self.X = self.data.drop("creditability", axis=1)
        self.y = self.data["creditability"]

        # Resampling using SMOTE to balance the dataset
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        self.X, self.y = smote.fit_resample(self.X, self.y)

        self.data = pd.concat([self.X, self.y], axis=1)

    def preprocess_adult(self):

        #Load data
        self.data = pd.read_csv(self.url, header=None, delimiter=',')
        self.data = self.data.map(lambda x: x.strip() if isinstance(x, str) else x)

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
        self.data['income'] = self.data['income'].map({'<=50K': 0, '>50K': 1})

        # Encode categorical variables using one-hot encoding
        categorical_cols = [
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"
        ]
        self.data = pd.get_dummies(self.data, columns=categorical_cols, drop_first=True)

        # Assign target & feature variables
        self.X = self.data.drop("income", axis=1)
        self.y = self.data["income"]

        # Resampling using SMOTE to balance the dataset
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        self.X, self.y = smote.fit_resample(self.X, self.y)

        self.data = pd.concat([self.X, self.y], axis=1)

    def preprocess_compass(self):
        # Load data
        self.data = pd.read_csv(self.url, header=0, delimiter=',')

        # Filter the dataset to only include the specified columns
        columns_to_keep = ['sex', 'age', 'race', 'juv_fel_count', 'juv_misd_count',
                           'juv_other_count', 'priors_count', 'c_charge_degree',
                           'two_year_recid']
        self.data = self.data[columns_to_keep]

        # One-hot encode categorical columns
        categorical_cols = ["sex", "race", "c_charge_degree"]
        self.data = pd.get_dummies(self.data, columns=categorical_cols, drop_first=True)

        # Assign target & feature variables
        self.X = self.data.drop("two_year_recid", axis=1)
        self.y = self.data["two_year_recid"]

        # Resampling using SMOTE to balance the dataset
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        self.X, self.y = smote.fit_resample(self.X, self.y)

        self.data = pd.concat([self.X, self.y], axis=1)