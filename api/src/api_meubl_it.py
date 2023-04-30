from fastapi import FastAPI, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
import logging
import numpy as np
import cv2
import base64

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


@app.post("/api_inference_pipeline")
async def main(payload: dict = Body(...)):
    encoded_img = payload["encoded_img"]

    LOGGER.info("Decode the image from base64")
    decoded_image = base64.b64decode(encoded_img)

    LOGGER.info("Convert the image in numpy array")
    img_bytes = np.frombuffer(decoded_image, np.uint8)

    LOGGER.info("Decode the image with OpenCV and convert to grayscale")
    img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)

    LOGGER.info("Convert the numpy image into bytes chain")
    _, encoded_gray_img = cv2.imencode('.png', img)
    bytes_gray_img = encoded_gray_img.tobytes()

    LOGGER.info("Encode the image as base64")
    encoded_bytes_gray_img = base64.b64encode(bytes_gray_img).decode("utf-8")

    LOGGER.info("Return the encoded image")
    return {
        "statusCode": 200,
        "body": encoded_bytes_gray_img,
        "isBase64Encoded": True,
        "headers": {"content-type": "application/json"}
    }


@app.post("/")
async def root(request: Request):
    json = await request.json()
    LOGGER.info(json)


handler = Mangum(app)
