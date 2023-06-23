################ Lambda ################

locals {
  lambda_permission_statement_id = "AllowExecutionFromS3"
}

resource "aws_lambda_function" "lambda_function_retrain_pipeline" {
  depends_on    = [null_resource.tensorflow_image]
  function_name = var.lambda_function_name_retrain_pipeline
  role          = aws_iam_role.iam_role_lambda_retrain_pipeline.arn
  image_uri     = "${aws_ecr_repository.ecr_repo_retrain_pipeline.repository_url}@${data.aws_ecr_image.ecr_tensorflow_image.id}"
  package_type  = "Image"
  memory_size   = var.lambda_function_memory_size_retrain_pipeline
  timeout       = var.lambda_function_timeout_retrain_pipeline
  vpc_config {
    security_group_ids = [var.sg_lambda_inference_pipeline_id]
    subnet_ids         = [var.private_subnet_inference_pipeline_id]
  }
  environment {
    variables = {
      BUCKET_NAME               = var.s3_meubl_it_bucket_name
      MODEL_KEY                 = var.s3_object_meubl_it_model_key
      ANNOTATED_DATA_FOLDER_KEY = var.s3_object_annotated_data_key
      TRAINED_DATA_FOLDER_KEY   = aws_s3_object.s3_object_trained.key
    }
  }
}

resource "aws_lambda_permission" "lambda_perm_api_gw_retrain_pipeline" {
  depends_on    = [aws_lambda_function.lambda_function_retrain_pipeline]
  statement_id  = local.lambda_permission_statement_id
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.lambda_function_retrain_pipeline.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = "arn:aws:s3:::${var.s3_meubl_it_bucket_name}"
}

resource "aws_s3_bucket_notification" "bucket_terraform_notification" {
  bucket = var.s3_meubl_it_bucket_name
  lambda_function {
    lambda_function_arn = aws_lambda_function.lambda_function_retrain_pipeline.arn
    events              = ["s3:ObjectCreated:*"]
    filter_suffix       = ".png"
    filter_prefix       = var.s3_object_annotated_data_key
  }
  depends_on = [aws_lambda_permission.lambda_perm_api_gw_retrain_pipeline]
}