from fastapi import FastAPI, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
import logging
import numpy as np
import cv2
import base64
import boto3
import os

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

# s3_endpoint_interface_url = os.environ['ENDPOINT_URL']
# s3_endpoint_interface_id = os.environ['ENDPOINT_ID']
# region = os.environ['REGION']

# client = boto3.client('ec2', region_name=region)
# response = client.describe_vpc_endpoints(VpcEndpointIds=[s3_endpoint_interface_id])
# s3_endpoint_url = response['VpcEndpoints'][0]['DnsEntries'][0]['DnsName']

# LOGGER.info(s3_endpoint_interface_id)
# s3 = boto3.client('s3', endpoint_url=s3_endpoint_url)


@app.post("/api_inference_pipeline")
async def main():
    LOGGER.info("TOTO")
    # bucket_name = 's3-bucket-inference-pipeline-esgi'
    # #Get the image from the S3 bucket
    # s3_file_to_download = 'provider_file'
    # object_data = s3.get_object(Bucket=bucket_name, Key=s3_file_to_download)
    # LOGGER.info("Fichier trouvé")
    # file_content = object_data['Body'].read().decode('utf-8')
    #
    # # Affichez le contenu du fichier
    # LOGGER.info(file_content)


# @app.post("/api_inference_pipeline")
# async def main(payload: dict = Body(...)):
#     encoded_img = payload["encoded_img"]
#
#     LOGGER.info("Decode the image from base64")
#     decoded_image = base64.b64decode(encoded_img)
#
#     LOGGER.info("Convert the image in numpy array")
#     img_bytes = np.frombuffer(decoded_image, np.uint8)
#
#     LOGGER.info("Decode the image with OpenCV and convert to grayscale")
#     img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
#
#     LOGGER.info("Convert the numpy image into bytes chain")
#     _, encoded_gray_img = cv2.imencode('.png', img)
#     bytes_gray_img = encoded_gray_img.tobytes()
#
#     LOGGER.info("Encode the image as base64")
#     encoded_bytes_gray_img = base64.b64encode(bytes_gray_img).decode("utf-8")
#
#     LOGGER.info("Return the encoded image")
#     return {
#         "statusCode": 200,
#         "body": encoded_bytes_gray_img,
#         "isBase64Encoded": True,
#         "headers": {"content-type": "application/json"}
#     }


@app.post("/")
async def root(request: Request):
    json = await request.json()
    LOGGER.info(json)


handler = Mangum(app)
