from kfp import dsl
from kfp.dsl import Dataset, Output

@dsl.component(
    packages_to_install=[
        "pandas==2.3.3",
        "boto3==1.43.6",
        "requests==2.33.1", # CVE-2026-25645
        "urllib3==2.7.0",
        "pygments==2.20.0" # CVE-2026-4539
    ],
    base_image="python:3.12",
)
def pull_data(
    matches_csv: Output[Dataset],
    s3_bucket: str = "mlops-bif",
    s3_key: str = "data/raw/dataset.csv",
    s3_endpoint: str = "https://b09595e7413d52541d55ebc0bf445b9c.r2.cloudflarestorage.com",
    s3_region: str = "auto",
):
    import os

    import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=os.environ["DVC_S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["DVC_S3_SECRET_KEY"],
        region_name=s3_region,
    )
    s3.download_file(s3_bucket, s3_key, matches_csv.path)
    print(f"Downloaded s3://{s3_bucket}/{s3_key} to {matches_csv.path}")
