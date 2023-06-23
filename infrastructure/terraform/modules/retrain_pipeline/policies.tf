################ Policies ################

data "aws_iam_policy_document" "iam_policy_lambda" {
  statement {
    sid     = ""
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "iam_role_lambda_retrain_pipeline" {
  name               = var.iam_role_lambda_name_retrain_pipeline
  assume_role_policy = data.aws_iam_policy_document.iam_policy_lambda.json
}

resource "aws_iam_role_policy_attachment" "iam_role_policy_attachment_lambda_vpc_access_execution" {
  role       = aws_iam_role.iam_role_lambda_retrain_pipeline.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
}

resource "aws_iam_role_policy_attachment" "iam_role_policy_attachment_lambda_s3_access_execution" {
  role       = aws_iam_role.iam_role_lambda_retrain_pipeline.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_role_policy_attachment" "process_logging_policy_attachment" {
  role       = aws_iam_role.iam_role_lambda_retrain_pipeline.name
  policy_arn = var.iam_policy_process_logging_policy_arn
}
