################## s3 #################
resource "aws_s3_bucket" "s3_bucket_inference_pipeline" {
  bucket = var.bucket_name_retrain_pipeline
}
