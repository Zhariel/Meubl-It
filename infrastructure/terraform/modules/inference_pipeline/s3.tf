# Define the S3 bucket for the inference pipeline
resource "aws_s3_bucket" "this" {
  //bucket = var.s3_function_name_inference_pipeline
  bucket = "s3-bucket-inference-pipeline-esgi"
}

resource "aws_iam_role" "iam_role_s3_inference_pipeline" {
  //name = var.iam_role_s3_name_inference_pipeline
  name = "iam-role-s3-inference-pipeline-esgi"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
      }
    ]
  })
}

/*resource "aws_s3_bucket_policy" "this" {
  bucket = aws_s3_bucket.this.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:*",
        ]
        Effect = "Allow"
        Resource = "${aws_s3_bucket.this.arn}*//*"
        Principal = "*"
      }
    ]
  })
}*/

resource "aws_s3_object" "file_upload" {
  bucket = aws_s3_bucket.this.id
  key    = "provider_file"
  force_destroy = true
  source = "/Users/franchouillard/Documents/GitHub/Meubl-It/infrastructure/terraform/modules/inference_pipeline/route_table.tf"
  etag = filemd5("/Users/franchouillard/Documents/GitHub/Meubl-It/infrastructure/terraform/modules/inference_pipeline/route_table.tf")
}
