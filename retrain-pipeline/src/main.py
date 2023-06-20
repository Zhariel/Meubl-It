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

s3 = boto3.resource('s3')

def handler(event, context):
    LOGGER.info("Event: " + str(event))
    LOGGER.info("Context: " + str(context))
    return {"statusCode": 200, "body": "Hello World"}

