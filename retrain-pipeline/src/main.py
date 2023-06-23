import logging
import numpy as np
import boto3
import os
import json
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
batch_size = 2


def load_from_s3(LOGGER, bucket_name, key_file):
    s3 = boto3.client('s3')
    LOGGER.info(f"Get {key_file} from {bucket_name} S3 bucket")
    response = s3.get_object(Bucket=bucket_name, Key=key_file)
    file_bytes = response['Body'].read()
    return file_bytes


def fake_retrain():
    LOGGER.info("Fake retrain")
    s3 = boto3.resource('s3')
    s3.Object(bucket_name, annotated_data_folder_key).copy_from(CopySource=f"{bucket_name}/{trained_data_folder_key}")
    s3.Object(bucket_name, annotated_data_folder_key).delete()
    return True


def handler(event, context):
    LOGGER.info("Event: " + str(event))

    s3 = boto3.client('s3')
    LOGGER.info(f"Get list of images from {bucket_name}/{annotated_data_folder_key} S3 bucket")
    response = s3.list_objects(Bucket=bucket_name, Prefix=annotated_data_folder_key)
    images = [obj['Key'] for obj in response['Contents']]
    LOGGER.info("List images: " + str(images))
    nb_data = (len(images) - 1) / 2
    if nb_data < batch_size:
        LOGGER.info(f"Number of images in {bucket_name}/{annotated_data_folder_key} S3 bucket is less than batch size")
        return {"statusCode": 200, "body": "Number of images in S3 bucket is less than batch size"}
    LOGGER.info(f"Number of images in {bucket_name}/{annotated_data_folder_key} S3 bucket is {nb_data}")

    box_array = []
    label_array = []
    images_array = []

    for i in range(1, len(images), 2):
        img_file = load_from_s3(LOGGER, bucket_name, images[i])
        image_stream = BytesIO(img_file)
        image = Image.open(image_stream).convert('RGB')

        metadata_file = load_from_s3(LOGGER, bucket_name, images[i + 1]).decode('utf-8')
        metadata = json.loads(metadata_file)
        LOGGER.info("metadata json: " + str(metadata))
        box = (
            int(float(metadata["start_x_axis"])),
            int(float(metadata["end_x_axis"])),
            int(float(metadata["start_y_axis"])),
            int(float(metadata["end_y_axis"]))
        )

        images_array.append(image)
        label_array.append(metadata["selected_furniture"])
        box_array.append(box)

    LOGGER.info("Box array: " + str(box_array))
    LOGGER.info("Label array: " + str(label_array))
    LOGGER.info("Images array: " + str(images_array))

    fake_retrain()

    return {"statusCode": 200, "body": "fake retrained"}
