import logging
import numpy as np
import base64
import boto3
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

bucket_name = os.environ['BUCKET_NAME']
model_key = os.environ['MODEL_KEY']
annotated_data_folder_key = os.environ['ANNOTATED_DATA_FOLDER_KEY']
trained_data_folder_key = os.environ['TRAINED_DATA_FOLDER_KEY']
batch_size = 3
folder_name = "annotated"

s3 = boto3.resource('s3')

def fake_retrain():
    LOGGER.info("Fake retrain")
    s3.Object(bucket_name, annotated_data_folder_key).copy_from(CopySource=f"{bucket_name}/{trained_data_folder_key}")
    s3.Object(bucket_name, annotated_data_folder_key).delete()
    return True
def handler(event, context):
    LOGGER.info("Event: " + str(event))
    LOGGER.info("Context: " + str(context))

    LOGGER.info(f"Get list of images from {bucket_name}/{folder_name} S3 bucket")
    response = s3.list_objects(Bucket=bucket_name, Prefix=folder_name)
    images = [obj['Key'] for obj in response['Contents']]

    if(len(images) < batch_size):
        LOGGER.info(f"Number of images in {bucket_name}/{folder_name} S3 bucket is less than batch size")
        return {"statusCode": 200, "body": "Number of images in S3 bucket is less than batch size"}

    LOGGER.info(f"Number of images in {bucket_name}/{folder_name} S3 bucket is {len(images)}")
    fake_retrain()

    return {"statusCode": 200, "body": "fake retrained"}