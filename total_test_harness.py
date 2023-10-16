from flipping_script import TotalTestMethod
from pathlib import Path

MODEL_DIRECTORY = r'models/'
PREDICTION_DIRECTORY = r'predictions/'
TOTAL_TEST_DIRECTORY = r'total_tests/'
NUMBER_OF_REPEATS = 1

file_base_names = []
model_file = Path(MODEL_DIRECTORY)
for e in model_file.iterdir():
    file_base_names.append(e.name)

for i in range(NUMBER_OF_REPEATS):
    for file_base_name in file_base_names:
        prediction_file = Path(
            f'{PREDICTION_DIRECTORY}{file_base_name}.csv')
        model_file = Path(f'{MODEL_DIRECTORY}{file_base_name}')
        generated_tests_file = Path(
            f'{TOTAL_TEST_DIRECTORY}{file_base_name}_{i}.csv')

        categorical_attributes = None

        if 'german' in model_file.name:
            sensitive_attribute = 'sex'
            categorical_attributes = [
                "checking_status", "credit_history", "purpose", "savings_status",
                "employment_duration", "other_debtors", "property",
                "other_installment_plans", "housing", "job", "phone",
                "foreign_worker"]
        elif 'adult' in model_file.name:
            sensitive_attribute = 'sex'
            categorical_attributes = [
                "workclass", "education", "marital-status", "occupation",
                "relationship", "race", "native-country"]
        elif 'compas' in model_file.name:
            sensitive_attribute = 'sex'
            categorical_attributes = ["race", "c_charge_degree"]

        print(f'\nmodel file: {model_file}, repeat:{i}\n')

        total_test = TotalTestMethod(
            predictions_file_path=prediction_file,
            model_file_path=model_file,
            sensitive_attribute=sensitive_attribute,
            predictive_attributes=categorical_attributes,
            target_variable='Predicted_Labels')

        total_test.run_total_tests()
        total_test.save_failed_cases_to_csv(generated_tests_file)
