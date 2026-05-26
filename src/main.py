import kfp  # type: ignore


def _create_client() -> kfp.Client:
    import os

    from auth import KFPClientManager  # type: ignore

    kfp_client_manager = KFPClientManager(
        api_url=os.environ["KUBEFLOW_URL"],
        dex_username=os.environ["KUBEFLOW_USER"],
        dex_password=os.environ["KUBEFLOW_PASSWORD"],
        dex_auth_type="local",
        namespace="csgo2",
    )

    kfp_client = kfp_client_manager.create_kfp_client()

    return kfp_client


def compile() -> None:
    from kfp import compiler  # type: ignore

    from pipeline import cs_go_match_predictor_train  # type: ignore

    compiler.Compiler().compile(cs_go_match_predictor_train, "pipeline.yaml")  # type: ignore


PIPELINE_NAME = "cs-go-match-predictor-train"


def _find_pipeline_id(kfp_client: kfp.Client, name: str) -> str | None:
    pid = kfp_client.get_pipeline_id(name)
    if pid:
        return pid
    for ns in ("csgo2", None):
        page_token = ""
        while True:
            kwargs = {"page_size": 100, "page_token": page_token}
            if ns:
                kwargs["namespace"] = ns
            try:
                resp = kfp_client.list_pipelines(**kwargs)
            except TypeError:
                resp = kfp_client.list_pipelines(page_size=100, page_token=page_token)
            pipelines = resp.pipelines or []
            print(f"[debug] ns={ns} found {len(pipelines)} pipelines on page")
            for p in pipelines:
                pname = getattr(p, "display_name", None) or getattr(p, "name", None)
                print(f"  - {pname!r} (id={p.pipeline_id})")
                if pname == name:
                    return p.pipeline_id
            page_token = resp.next_page_token or ""
            if not page_token:
                break
    return None


def upload() -> None:
    import time

    from dotenv import load_dotenv
    from kfp_server_api.exceptions import ApiException

    load_dotenv()
    kfp_client = _create_client()

    version_name = f"v-{time.strftime('%Y%m%d-%H%M%S')}"
    pipeline_id = _find_pipeline_id(kfp_client, PIPELINE_NAME)

    if pipeline_id is None:
        try:
            result = kfp_client.upload_pipeline(
                "pipeline.yaml",
                pipeline_name=PIPELINE_NAME,
                namespace="csgo2",
            )
            print(f"Created pipeline {PIPELINE_NAME} ({result.pipeline_id})")
            return
        except ApiException as e:
            if e.status != 409:
                raise
            print("Pipeline already exists (409), looking it up again...")
            pipeline_id = _find_pipeline_id(kfp_client, PIPELINE_NAME)
            if pipeline_id is None:
                raise RuntimeError(
                    "Got 409 on upload but cannot find the existing pipeline. "
                    "Check the KFP UI for the actual name."
                ) from e

    result = kfp_client.upload_pipeline_version(
        pipeline_package_path="pipeline.yaml",
        pipeline_version_name=version_name,
        pipeline_id=pipeline_id,
    )
    print(f"Uploaded version {version_name} on pipeline {pipeline_id}")


def run() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    kfp_client = _create_client()

    run = kfp_client.create_run_from_pipeline_package(
        "pipeline.yaml",
        arguments={},
        namespace="csgo2",
    )

    print(run.run_id)


def run_existing() -> None:
    """Lance un run sur une pipeline déjà uploadée (pas de recompile).

    Usage:
        PIPELINE_ID=... [PIPELINE_VERSION_ID=...] [EXPERIMENT_NAME=Default] \
        uv run python -c "from main import run_existing; run_existing()"

    Les hyperparams peuvent être surchargés via env vars préfixées ARG_,
    ex: ARG_learning_rate=0.005 ARG_epochs=20
    """
    import json
    import os
    import time

    from dotenv import load_dotenv

    load_dotenv()
    kfp_client = _create_client()

    pipeline_id = os.environ["PIPELINE_ID"]
    version_id = os.environ.get("PIPELINE_VERSION_ID")
    experiment_name = os.environ.get("EXPERIMENT_NAME", "Default")
    run_name = os.environ.get("RUN_NAME", f"run-{int(time.time())}")

    if not version_id:
        versions = kfp_client.list_pipeline_versions(pipeline_id=pipeline_id, page_size=1, sort_by="created_at desc")
        version_id = versions.pipeline_versions[0].pipeline_version_id
        print(f"Using latest version: {version_id}")

    experiment = kfp_client.create_experiment(name=experiment_name, namespace="csgo2")

    arguments = {}
    for k, v in os.environ.items():
        if not k.startswith("ARG_"):
            continue
        name = k[len("ARG_"):]
        try:
            arguments[name] = json.loads(v)
        except json.JSONDecodeError:
            arguments[name] = v
    print(f"Arguments: {arguments}")

    run = kfp_client.run_pipeline(
        experiment_id=experiment.experiment_id,
        job_name=run_name,
        pipeline_id=pipeline_id,
        version_id=version_id,
        params=arguments,
    )
    print(f"Run ID: {run.run_id}")
    print(f"URL: {os.environ['KUBEFLOW_URL']}/pipeline/#/runs/details/{run.run_id}")


if __name__ == "__main__":
    compile()

    upload()

    run()