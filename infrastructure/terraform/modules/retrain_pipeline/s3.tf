################## s3 #################

resource "aws_s3_object" "s3_object_trained" {
  bucket = var.s3_bucket_retrain_pipeline_id
  force_destroy = true
  key    = "trained/"
  acl    = "private"
}
