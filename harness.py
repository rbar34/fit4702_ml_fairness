from pipeline import BaseModelPipeline, RandomForestPipeline, SVMPipeline, LogisticPipeline, NeuralNetworkPipeline
import tensorflow as tf
tf.keras.utils.disable_interactive_logging()

"""
The group that stands to be harmed or disadvantaged more by an incorrect positive or negative prediction should guide the
choice of positive class.

:Labels:

German credit: 1 (Good credit), 0 (Bad credit)

Compos: 0 (Low recid), 1 (high recid)

Adult income: 0 (low Income), 1 (high income) 
"""

# Define data path
datasets = {
    "german": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
    "compas": "compas.csv",
    "adult": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
}

# Different tests to be ran
pipelines = {
    "LOGISTIC": LogisticPipeline,
    "NEURAL NET": NeuralNetworkPipeline,
    "RANDOM FOREST": RandomForestPipeline,
    "SVM": SVMPipeline,
}

# Execute pipeline
for pipeline_name, PipelineClass in pipelines.items():
    print(f'\n----------------{pipeline_name}-----------------')
    for dataset_name, dataset_url in datasets.items():
        print(f"\nProcessing {dataset_name} with {pipeline_name}")
        pipeline = PipelineClass(dataset_url)
        pipeline.run_pipeline()
        predictions_filename = f"./predictions/{dataset_name}_{pipeline_name}.csv".replace(" ", "_").lower()
        pipeline.save_predictions_to_csv(predictions_filename)
        model_filename = f"./models/{dataset_name}_{pipeline_name}".replace(" ", "_").lower()
        pipeline.save_model(model_filename)
