################## s3 #################

resource "aws_s3_object" "s3_object_trained" {
  bucket        = var.s3_meubl_it_bucket_id
  force_destroy = true
  key           = "trained/"
  acl           = "private"
}
