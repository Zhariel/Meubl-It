from fastapi import FastAPI, Request, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
import logging
import numpy as np
import cv2
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

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bucket_name = os.environ['BUCKET_NAME']
model_key = os.environ['MODEL_KEY']

s3 = boto3.resource('s3')


class Sample(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.acti = nn.ReLU()
        self.output = nn.Conv2d(32, 3, kernel_size=(3, 3), stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input(x)
        x = self.acti(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


def load_model_from_s3(bucket_name, model_key):
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket_name, Key=model_key)
    model_bytes = response['Body'].read()
    model_file = BytesIO(model_bytes)
    LOGGER.info("Load model from s3 bucket")
    model = Sample()
    LOGGER.info("Load model state dict")
    model.load_state_dict(torch.load(model_file))
    LOGGER.info("Model eval")
    model.eval()
    return model


def inference(input_image, model):
    preprocess = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    input_tensor = preprocess(input_image).unsqueeze(0)

    output_tensor = model(input_tensor)
    output_tensor = output_tensor.squeeze(0)
    output_tensor = output_tensor.permute(1, 2, 0)
    output_tensor = output_tensor.detach().cpu().numpy()
    LOGGER.info("Convert output tensor to image")
    output_tensor = (output_tensor * 255).astype(np.uint8)
    output_image = Image.fromarray(output_tensor)
    return output_image


@app.post("/inference_pipeline")
async def main(payload: dict = Body(...)):
    LOGGER.info("Start inference pipeline")

    LOGGER.info("Read model in s3 bucket")
    model = load_model_from_s3(bucket_name, model_key)

    encoded_img = payload["encoded_img"]
    selected_furniture = payload["selected_furniture"]
    start_x_axis = payload["start-x-axis"]
    end_x_axis = payload["end-x-axis"]
    start_y_axis = payload["start-y-axis"]
    end_y_axis = payload["end-y-axis"]
    LOGGER.info("Selected furniture: " + selected_furniture)
    LOGGER.info("start_x_axis: " + str(start_x_axis))
    LOGGER.info("end_x_axis: " + str(end_x_axis))
    LOGGER.info("start_y_axis: " + str(start_y_axis))
    LOGGER.info("end_y_axis: " + str(end_y_axis))

    LOGGER.info("Save encoded image in /unannotated S3 bucket")
    s3.Object(bucket_name, 'unannotated/' + str(np.random.randint(100000)) + '.png').put(Body=base64.b64decode(encoded_img))

    LOGGER.info("Decode the image from base64")
    decoded_image = base64.b64decode(encoded_img)

    LOGGER.info("Convert the image in data stream")
    image_stream = BytesIO(decoded_image)
    image = Image.open(image_stream).convert('RGB')

    LOGGER.info("Inference")
    output_image = inference(image, model)
    LOGGER.info("Save output image in /tmp")

    output_image_bytes = BytesIO()
    output_image.save(output_image_bytes, format='PNG')
    output_image_base64 = base64.b64encode(output_image_bytes.getvalue()).decode('utf-8')

    return {
        "statusCode": 200,
        "body": output_image_base64,
        "isBase64Encoded": True,
        "headers": {"content-type": "application/json"}
    }

@app.post("/")
async def root(request: Request):
    json = await request.json()
    LOGGER.info(json)

handler = Mangum(app)