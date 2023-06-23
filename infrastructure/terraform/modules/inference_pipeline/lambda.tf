################ Lambda ################

locals {
  lambda_permission_statement_id_inference_pipeline = "AllowExecutionFromAPIGatewayInferencePipeline"
  lambda_permission_statement_id_valid_captcha = "AllowExecutionFromAPIGatewayValidCaptcha"
  lambda_permission_statement_id_get_captcha = "AllowExecutionFromAPIGatewayGetCaptcha"
}

resource "aws_lambda_function" "lambda_function_inference_pipeline" {
  depends_on    = [null_resource.tensorflow_image]
  function_name = var.lambda_function_name_inference_pipeline
  role          = aws_iam_role.iam_role_lambda_inference_pipeline.arn
  image_uri     = "${aws_ecr_repository.ecr_repo_inference_pipeline.repository_url}@${data.aws_ecr_image.ecr_tensorflow_image.id}"
  package_type  = "Image"
  memory_size   = var.lambda_function_memory_size_inference_pipeline
  timeout       = var.lambda_function_timeout_inference_pipeline
  vpc_config {
    security_group_ids = [aws_security_group.sg_lambda_inference_pipeline.id]
    subnet_ids         = [aws_subnet.private_subnet_inference_pipeline.id]
  }
  environment {
    variables = {
      BUCKET_NAME                 = var.bucket_name_inference_pipeline
      MODEL_KEY                   = var.bucket_key_model_inference_pipeline
      UNANNOTATED_DATA_FOLDER_KEY = aws_s3_object.s3_object_unannotated.key
      ANNOTATED_DATA_FOLDER_KEY   = aws_s3_object.s3_object_annotated.key
    }
  }
}

resource "aws_lambda_permission" "lambda_perm_api_gw_inference_pipeline" {
  statement_id  = local.lambda_permission_statement_id_inference_pipeline
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.lambda_function_inference_pipeline.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.api_gw_inference_pipeline.execution_arn}/*/${aws_api_gateway_method.method_api_gw_inference_pipeline.http_method}${aws_api_gateway_resource.resource_api_gw_inference_pipeline.path}"
}

resource "aws_lambda_permission" "lambda_perm_api_gw_valid_captcha" {
  statement_id  = local.lambda_permission_statement_id_valid_captcha
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.lambda_function_inference_pipeline.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.api_gw_inference_pipeline.execution_arn}/*/${aws_api_gateway_method.method_api_gw_valid_captcha.http_method}${aws_api_gateway_resource.resource_api_gw_valid_captcha.path}"
}

resource "aws_lambda_permission" "lambda_perm_api_gw_get_captcha" {
  statement_id  = local.lambda_permission_statement_id_get_captcha
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.lambda_function_inference_pipeline.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.api_gw_inference_pipeline.execution_arn}/*/${aws_api_gateway_method.method_api_gw_get_captcha.http_method}${aws_api_gateway_resource.resource_api_gw_get_captcha.path}"
}
