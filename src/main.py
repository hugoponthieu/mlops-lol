import kfp  # type: ignore


def _create_client() -> kfp.Client:
    import os

    from auth import KFPClientManager  # type: ignore

    kfp_client_manager = KFPClientManager(
        api_url=os.environ["KUBEFLOW_URL"],
        dex_username=os.environ["KUBEFLOW_USER"],
        dex_password=os.environ["KUBEFLOW_PASSWORD"],
        dex_auth_type="local",
    )

    kfp_client = kfp_client_manager.create_kfp_client()

    return kfp_client


def compile() -> None:
    from kfp import compiler  # type: ignore

    from pipeline import cs_go_match_predictor_train  # type: ignore

    compiler.Compiler().compile(cs_go_match_predictor_train, "pipeline.yaml")  # type: ignore


def upload() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    kfp_client = _create_client()

    kfp_client.upload_pipeline(
        "pipeline.yaml",
        namespace="kubeflow-gomez",
    )


def run() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    kfp_client = _create_client()

    run = kfp_client.create_run_from_pipeline_package(
        "pipeline.yaml",
        arguments={},
        namespace="kubeflow-gomez",
    )

    print(run.run_id)


if __name__ == "__main__":
    compile()

    upload()

    run()