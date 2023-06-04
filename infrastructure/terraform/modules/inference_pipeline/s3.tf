################ S3 ################
# S3 bucket for model
resource "aws_s3_bucket" "s3_bucket_inference_pipeline" {
  bucket = var.bucket_name_inference_pipeline
}

resource "aws_s3_object" "s3_object_inference_pipeline" {
  bucket        = aws_s3_bucket.s3_bucket_inference_pipeline.id
  key           = var.bucket_key_model_inference_pipeline
  force_destroy = true
  source        = var.model_path_inference_pipeline
  etag          = filesha256(var.model_path_inference_pipeline)
}
# same s3 bucket for data
resource "aws_s3_bucket" "s3_bucket_inference_pipeline_data" {
  bucket = var.bucket_name_inference_pipeline
}

resource "aws_s3_object" "s3_object_annotated" {
  bucket = aws_s3_bucket.s3_bucket_inference_pipeline_data.id
  force_destroy = true
  key    = "annotated/"
  acl    = "private"
}

resource "aws_s3_object" "s3_object_unannotated" {
  bucket = aws_s3_bucket.s3_bucket_inference_pipeline_data.id
  force_destroy = true
  key    = "unannotated/"
  acl    = "private"
}