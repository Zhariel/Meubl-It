import logging
import numpy as np
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

s3 = boto3.resource('s3')


def fake_retrain():
    LOGGER.info("Fake retrain")
    s3.Object(bucket_name, annotated_data_folder_key).copy_from(CopySource=f"{bucket_name}/{trained_data_folder_key}")
    s3.Object(bucket_name, annotated_data_folder_key).delete()
    return True


def handler(event, context):
    LOGGER.info("Event: " + str(event))
    LOGGER.info("Context: " + str(context))

    LOGGER.info(f"Get list of images from {bucket_name}/{annotated_data_folder_key} S3 bucket")
    response = s3.list_objects(Bucket=bucket_name, Prefix=annotated_data_folder_key)
    images = [obj['Key'] for obj in response['Contents']]
    images_data = [s3.get_object(Bucket=bucket_name, Key=obj['Key'])['Body'].read() for obj in response['Contents']]

    if len(images) < batch_size:
        LOGGER.info(f"Number of images in {bucket_name}/{annotated_data_folder_key} S3 bucket is less than batch size")
        return {"statusCode": 200, "body": "Number of images in S3 bucket is less than batch size"}

    LOGGER.info(f"Number of images in {bucket_name}/{annotated_data_folder_key} S3 bucket is {len(images)}")

    fake_retrain()

    box_array = []
    label_array = []
    images_array = []
    # for box in images:
    #     box_array.append(box)
    # for label in images:
    #     label_array.append(label)
    # for image in images:
    #     images_array.append(image)
    for box, label, image in zip(images):
        box_array.append(box)
        label_array.append(label)
        images_array.append(image)
    LOGGER.info("Box array: " + str(box_array))
    LOGGER.info("Label array: " + str(label_array))
    LOGGER.info("Images array: " + str(images_array))


    LOGGER.info("List images: " + str(images))

    return {"statusCode": 200, "body": "fake retrained"}
