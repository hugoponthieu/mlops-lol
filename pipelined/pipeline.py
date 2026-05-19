
@pipeline()
def data_pipeline():
    step1 = my_components.dvc_pull()
    step2 = my_components.feature_engineering(input_results_dataset=step1.output["results_dataset"])
    step3 = my_components.preprocessing(featured_results_dataset=step2.output["featured_results_dataset"])
    step4 = my_components.train_model(input_preprocessed_train_dataset=step3.output["train_data_set"])
    step5 = my_components.evaluate_model(input_preprocessed_dataset=step3.output["test_data_set"], input_raw_dataset=step1.output["results_dataset"], input_trained_dataset=step4.output["trained_results_dataset"], input_keras_model=step4.output["output_model_keras"], input_preprocessing_joblib=step3.output["preprocessing_joblib"])

if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        pipeline_func=arithmetic_pipeline, 
        package_path="my_pipeline.yaml"
    )
