from io import BytesIO
import random
import base64
import json
import zlib

from utils import load_from_s3

import boto3


def load_captcha(LOGGER, bucket_name, folder_name):
    s3 = boto3.client('s3')
    LOGGER.info(f"Get list of images from {bucket_name}/{folder_name} S3 bucket")
    response = s3.list_objects(Bucket=bucket_name, Prefix=folder_name)
    images = [obj['Key'] for obj in response['Contents']]

    LOGGER.info("Choice a random image from the list of images")
    random_image_key = random.choice(images)

    LOGGER.info(f"Get the random image from {bucket_name} S3 bucket")
    img_file = load_from_s3(LOGGER, bucket_name, random_image_key)

    LOGGER.info("Compress the image to the captcha")
    output_image_compress = zlib.compress(img_file.getvalue())

    LOGGER.info("Encode the image to the captcha in base64")
    output_image_base64 = base64.b64encode(output_image_compress).decode('utf-8')

    return output_image_base64, random_image_key


def save_captcha_data(LOGGER, bucket_name, key_folder, key_img_captcha, selected_furniture, start_x_axis,
                      end_x_axis, start_y_axis, end_y_axis):
    LOGGER.info(f"Get the captcha image {key_img_captcha} from {bucket_name} S3 bucket")
    img_captcha_file = load_from_s3(LOGGER, bucket_name, key_img_captcha)

    key_img = key_img_captcha.split("/")[1]
    s3 = boto3.client('s3')
    LOGGER.info(f"Save captcha image in {key_folder} folder of {bucket_name} S3 bucket")
    s3.put_object(
        Bucket=bucket_name,
        Key=f'{key_folder}{key_img.split(".")[0]}/img.{key_img.split(".")[1]}',
        Body=img_captcha_file
    )

    metadata = json.dumps({
        'selected_furniture': selected_furniture,
        'start_x_axis': start_x_axis,
        'end_x_axis': end_x_axis,
        'start_y_axis': start_y_axis,
        'end_y_axis': end_y_axis
    })

    LOGGER.info(f"Save metadata captcha in {key_folder} folder of {bucket_name} S3 bucket")
    s3.put_object(
        Bucket=bucket_name,
        Key=f'{key_folder}{key_img.split(".")[0]}/metadata.json',
        Body=metadata
    )

    LOGGER.info(f"Delete the captcha image in {key_img_captcha.split('/')[0]}/ folder of {bucket_name} S3 bucket")
    s3.delete_object(Bucket=bucket_name, Key=key_img_captcha)
