import boto3
import os
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
DEFAULT_REGION = os.getenv('DEFAULT_REGION')

s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=DEFAULT_REGION
)

bucket_name = os.getenv('BUCKET_NAME')
prefix = os.getenv('PREFIX')
local_dir = './downloads'
limit = 2000

os.makedirs(local_dir, exist_ok=True)

count = 0

paginator = s3.get_paginator('list_objects_v2')

for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
    for obj in page.get('Contents', []):
        key = obj['Key']

        if key.endswith('.json'):
            local_path = os.path.join(local_dir, os.path.basename(key))

            print(f"{count} - Downloading {key}")
            s3.download_file(bucket_name, key, local_path)
            count += 1
            if count >= limit:
                print("✅ Reached 500 files")
                break

    if count >= limit:
        break