################ Cloudwatch ################

resource "aws_cloudwatch_log_group" "logs_lambda_inference_pipeline" {
  name              = "/aws/lambda/${aws_lambda_function.lambda_function_inference_pipeline.function_name}"
  retention_in_days = 7
  lifecycle {
    prevent_destroy = false
  }
}
