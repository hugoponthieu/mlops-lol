from kfp import dsl
from kfp.dsl import Dataset, Input, Output

@dsl.component(
    packages_to_install=[
        "pandas==3.0.2",
        "dvc[s3]==3.67.1",
        "python-dotenv==1.2.2",
        "gitpython==3.1.50", # CVE-2026-42215, 42284, 44243, 44244, GHSA-mv93-w799-cj2w
        "aiohttp==3.13.5", # CVE-2026-22815, 34515, 34516, 34525
        "cryptography==48.0.0", # CVE-2026-39892
        "pygments==2.20.0" # CVE-2026-4539
    ],
    base_image="python:3.14",
)
def pull_data(
    matches_csv: Output[Dataset],
):
    from dvc.api import DVCFileSystem
    import os
    import pandas as pd

    repo = DVCFileSystem(
        os.environ["DVC_GIT_REPO"],
        remote="s3-storage",
        remote_config={
            "access_key_id": os.environ["DVC_S3_ACCESS_KEY"],
            "secret_access_key": os.environ["DVC_S3_SECRET_KEY"],
        },
    )

    with repo.open("data/raw/dataset.csv") as f:
        df = pd.read_csv(f)

    df.to_csv(matches_csv.path)