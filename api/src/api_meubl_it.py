import logging
import os

from inference import inference_model
from captcha import load_captcha, save_captcha_data

from fastapi import FastAPI, Request, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

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

BUCKET_NAME = os.environ['BUCKET_NAME']
MODEL_KEY = os.environ['MODEL_KEY']
UNANNOTATED_DATA_FOLDER_KEY = os.environ['UNANNOTATED_DATA_FOLDER_KEY']
ANNOTATED_DATA_FOLDER_KEY = os.environ['ANNOTATED_DATA_FOLDER_KEY']


@app.post("/inference_pipeline")
async def inference_pipeline(payload: dict = Body(...)):
    LOGGER.info("Start inference pipeline")

    encoded_img = payload["encoded_img"]
    selected_furniture = payload["selected_furniture"]
    start_x_axis = payload["start-x-axis"]
    end_x_axis = payload["end-x-axis"]
    start_y_axis = payload["start-y-axis"]
    end_y_axis = payload["end-y-axis"]

    output_image_base64 = inference_model(LOGGER, encoded_img, selected_furniture, start_x_axis, end_x_axis,
                                          start_y_axis, end_y_axis, BUCKET_NAME, MODEL_KEY, UNANNOTATED_DATA_FOLDER_KEY)

    LOGGER.info("End inference pipeline")
    return {
        "statusCode": 200,
        "body": output_image_base64,
        "isBase64Encoded": True,
        "headers": {"content-type": "application/json"}
    }


@app.post("/valid_captcha")
async def captcha(payload: dict = Body(...)):
    key_img_captcha = payload["key_img_captcha"]
    selected_furniture = payload["selected_furniture"]
    start_x_axis = payload["start-x-axis"]
    end_x_axis = payload["end-x-axis"]
    start_y_axis = payload["start-y-axis"]
    end_y_axis = payload["end-y-axis"]

    save_captcha_data(LOGGER, BUCKET_NAME, ANNOTATED_DATA_FOLDER_KEY, key_img_captcha, selected_furniture,
                      start_x_axis, end_x_axis, start_y_axis, end_y_axis)

    return {
        "statusCode": 200,
        "body": "true",
        "headers": {"content-type": "application/json"}
    }


@app.post("/get_captcha")
async def main():
    img_captcha, key_img_captcha = load_captcha(LOGGER, BUCKET_NAME, UNANNOTATED_DATA_FOLDER_KEY)
    return {
        "statusCode": 200,
        "body": {"img_captcha": img_captcha, "key_img_captcha": key_img_captcha},
        "isBase64Encoded": True,
        "headers": {"content-type": "application/json"}
    }


handler = Mangum(app)
