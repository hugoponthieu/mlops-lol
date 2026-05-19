from kfp.dsl import component, Output, Dataset

@component(
    base_image="python:3.14",
    packages_to_install=["pandas==3.01", "os", "dvc"],
)
def dvc_pull(results_dataset: Output[Dataset]):
    from dvc.api import DVCFileSystem  # type: ignore

    import pandas as pd #type: ignore

    repo = DVCFileSystem(
        "https://github.com/hugoponthieu/mlops-lol.git",
        rev="feat/teacher-notebooks"
    )

    with repo.open(results_dataset.path) as f:
        df = pd.read_csv(f)

    df.to_csv(f"{results_dataset.path}/raw/results.csv")

