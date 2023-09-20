from pipeline import BaseModelPipeline, RandomForestPipeline, SVMPipeline, LogisticPipeline, NeuralNetworkPipeline


datasets = {
    "german": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
    "communities": "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data",
    "adult": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
}

pipelines = {
    "NEURAL NET": NeuralNetworkPipeline,
    "RANDOM FOREST": RandomForestPipeline,
    "SVM": SVMPipeline,
    "LOGISTIC": LogisticPipeline,
}

for pipeline_name, PipelineClass in pipelines.items():
    print(f'\n{pipeline_name}')
    for dataset_name, dataset_url in datasets.items():
        print(f"Processing {dataset_name} with {pipeline_name}")
        pipeline = PipelineClass(dataset_url)
        pipeline.run_pipeline()
        filename = f"{dataset_name}_{pipeline_name}.csv".replace(" ", "_").lower()
        pipeline.save_predictions_to_csv(filename)