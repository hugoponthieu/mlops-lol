# MLOps project

## DVC : Use a remote storage to store data

```bash
dvc init
dvc add data/raw/dataset.csv
dvc remote add -d s3-storage s3://mlops-bif
dvc remote modify s3-storage endpointurl https://b09595e7413d52541d55ebc0bf445b9c.r2.cloudflarestorage.com/mlops-bif
dvc remote modify --local s3-storage access_key_id 'YOUR_S3_KEY'
dvc remote modify --local s3-storage secret_access_key 'YOUR_S3_SECRET_KEY'
dvc push
```

## Mise : Use differents Python environment

For each projects:

```bash
mise trust
```

```bash
mise install
```

```bash
uv pip install '.[dev]'
```

## Environment variables

Create a `.env` file in the root of the project from the `.env.template` file and fill in the values.

## Usage

### Python scripts

To generate the pipeline:

```bash
uv pip install '.[dev]'
uv run compile
```

### Kubernetes secrets

To create a Kubernetes secret from the `.env` file:

```bash
kubectl create secret generic dvc-credentials -n csgo3 --from-env-file=.env
```
