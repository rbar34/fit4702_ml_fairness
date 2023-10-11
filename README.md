# Evaluating Machine Learning Fairness Testing Approaches and Fairness Metrics

## Installation

1. Create a python virtual environment, then activate it
2. Clone this repo
3. Dependencies can be installed using either poetry or pip:

`poetry install`

`pip install -r requirements.txt`

## Usage

1. Run `harness.py`
    - This downloads the data sources, trains the models and then produces predictions on test data
    - Predictions are located in `./predictions/`
    - Serialised models are located in `./models/`
2. Run `random_test_harness.py`
    - This randomly generates pairs of inputs representing identical individuals, differing only on the value of the sensitive attribute
    - These inputs are saved in `./generated_tests/` if changing the sensitive variable results in a different classification
3. Run `test_aq.py`
    - This uses AEQUITAS to produce group fairness metrics on the predictions and the randomly generated test cases
    - The group fairness results are located in `./aequitas/` and `./random_generated_group_fairness/`
