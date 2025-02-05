import argparse
import yaml
import boto3
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="params.yaml")
    return parser.parse_args()

def download_from_minio(config):
    minio_config = config['data']
    session = boto3.session.Session()
    s3 = session.client(
        service_name='s3',
        endpoint_url=minio_config['minio_endpoint'],
        aws_access_key_id=os.environ.get('MINIO_ACCESS_KEY'),
        aws_secret_access_key=os.environ.get('MINIO_SECRET_KEY')
    )
    os.makedirs(os.path.dirname(minio_config['local_path']), exist_ok=True)
    s3.download_file(minio_config['bucket'], minio_config['object_key'], minio_config['local_path'])
    print(f"Downloaded data to {minio_config['local_path']}")

def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    download_from_minio(config)

if __name__ == "__main__":
    main()
