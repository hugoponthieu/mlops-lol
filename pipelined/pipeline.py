
@pipeline()
def data_pipeline():
    step1 = my_components.dvc_pull()
    step2 = my_components.feature_engineering(step1.output)
    step3 = my_components.preprocessing(step1.output)
    step4 = my_components.train_model(step1.output)
    step5 = my_components.evaluate_model(#TODO)

if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        pipeline_func=arithmetic_pipeline, 
        package_path="my_pipeline.yaml"
    )
