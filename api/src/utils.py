from io import BytesIO

import boto3


def load_from_s3(LOGGER, bucket_name, key_file):
    s3 = boto3.client('s3')
    LOGGER.info(f"Get {key_file} from {bucket_name} S3 bucket")
    response = s3.get_object(Bucket=bucket_name, Key=key_file)
    file_bytes = response['Body'].read()
    file = BytesIO(file_bytes)
    return file


