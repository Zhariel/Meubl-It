################ S3 ################
resource "aws_s3_bucket" "s3_meubl_it_bucket" {
  bucket = var.s3_meubl_it_bucket_name
}

resource "aws_s3_object" "s3_object_inference_pipeline" {
  depends_on    = [aws_s3_bucket.s3_meubl_it_bucket]
  bucket        = aws_s3_bucket.s3_meubl_it_bucket.id
  key           = var.s3_object_meubl_it_model_key
  force_destroy = true
  source        = var.meubl_it_model_path
  etag          = filesha256(var.meubl_it_model_path)
}

resource "aws_s3_object" "s3_object_annotated" {
  bucket        = aws_s3_bucket.s3_meubl_it_bucket.id
  force_destroy = true
  key           = "annotated/"
  acl           = "private"
}

resource "aws_s3_object" "s3_object_unannotated" {
  bucket        = aws_s3_bucket.s3_meubl_it_bucket.id
  force_destroy = true
  key           = "unannotated/"
  acl           = "private"
}
