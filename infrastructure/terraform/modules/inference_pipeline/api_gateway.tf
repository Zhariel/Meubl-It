################ API Gateway ################

resource "aws_api_gateway_rest_api" "api_gw_inference_pipeline" {
  name        = var.api_name_inference_pipeline
  description = var.api_description_inference_pipeline

  binary_media_types = [
    "*/*"
  ]
}

# Inference Pipeline
resource "aws_api_gateway_resource" "resource_api_gw_inference_pipeline" {
  depends_on = [
    aws_api_gateway_rest_api.api_gw_inference_pipeline
  ]
  rest_api_id = aws_api_gateway_rest_api.api_gw_inference_pipeline.id
  parent_id   = aws_api_gateway_rest_api.api_gw_inference_pipeline.root_resource_id
  path_part   = "inference_pipeline"
}

resource "aws_api_gateway_method" "method_api_gw_inference_pipeline" {
  depends_on = [
    aws_api_gateway_resource.resource_api_gw_inference_pipeline
  ]
  rest_api_id   = aws_api_gateway_rest_api.api_gw_inference_pipeline.id
  resource_id   = aws_api_gateway_resource.resource_api_gw_inference_pipeline.id
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_method_response" "method_response_api_gw_inference_pipeline" {
  depends_on = [
    aws_api_gateway_method.method_api_gw_inference_pipeline
  ]
  rest_api_id     = aws_api_gateway_rest_api.api_gw_inference_pipeline.id
  resource_id     = aws_api_gateway_resource.resource_api_gw_inference_pipeline.id
  http_method     = aws_api_gateway_method.method_api_gw_inference_pipeline.http_method
  status_code     = "200"
  response_models = {
    "application/json" = "Empty"
  }
}

resource "aws_api_gateway_integration" "integration_api_gw_inference_pipeline" {
  depends_on = [
    aws_api_gateway_method.method_api_gw_inference_pipeline
  ]
  rest_api_id             = aws_api_gateway_rest_api.api_gw_inference_pipeline.id
  resource_id             = aws_api_gateway_resource.resource_api_gw_inference_pipeline.id
  http_method             = aws_api_gateway_method.method_api_gw_inference_pipeline.http_method
  type                    = "AWS_PROXY"
  integration_http_method = "POST"
  uri                     = aws_lambda_function.lambda_function_inference_pipeline.invoke_arn
}

resource "aws_api_gateway_integration_response" "integration_response_api_gw_inference_pipeline" {
  depends_on = [
    aws_api_gateway_integration.integration_api_gw_inference_pipeline
  ]
  rest_api_id = aws_api_gateway_rest_api.api_gw_inference_pipeline.id
  resource_id = aws_api_gateway_resource.resource_api_gw_inference_pipeline.id
  http_method = aws_api_gateway_method.method_api_gw_inference_pipeline.http_method
  status_code = aws_api_gateway_method_response.method_response_api_gw_inference_pipeline.status_code

  response_templates = {
    "application/json" = "Empty"
  }
}

# Valid Captcha
resource "aws_api_gateway_resource" "resource_api_gw_valid_captcha" {
  depends_on = [
    aws_api_gateway_rest_api.api_gw_inference_pipeline
  ]
  rest_api_id = aws_api_gateway_rest_api.api_gw_inference_pipeline.id
  parent_id   = aws_api_gateway_rest_api.api_gw_inference_pipeline.root_resource_id
  path_part   = "valid_captcha"
}

resource "aws_api_gateway_method" "method_api_gw_valid_captcha" {
  depends_on = [
    aws_api_gateway_resource.resource_api_gw_valid_captcha
  ]
  rest_api_id   = aws_api_gateway_rest_api.api_gw_inference_pipeline.id
  resource_id   = aws_api_gateway_resource.resource_api_gw_valid_captcha.id
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_method_response" "method_response_api_gw_valid_captcha" {
  depends_on = [
    aws_api_gateway_method.method_api_gw_valid_captcha
  ]
  rest_api_id     = aws_api_gateway_rest_api.api_gw_inference_pipeline.id
  resource_id     = aws_api_gateway_resource.resource_api_gw_valid_captcha.id
  http_method     = aws_api_gateway_method.method_api_gw_valid_captcha.http_method
  status_code     = "200"
  response_models = {
    "application/json" = "Empty"
  }
}

resource "aws_api_gateway_integration" "integration_api_gw_valid_captcha" {
  depends_on = [
    aws_api_gateway_method.method_api_gw_valid_captcha
  ]
  rest_api_id             = aws_api_gateway_rest_api.api_gw_inference_pipeline.id
  resource_id             = aws_api_gateway_resource.resource_api_gw_valid_captcha.id
  http_method             = aws_api_gateway_method.method_api_gw_valid_captcha.http_method
  type                    = "AWS_PROXY"
  integration_http_method = "POST"
  uri                     = aws_lambda_function.lambda_function_inference_pipeline.invoke_arn
}

resource "aws_api_gateway_integration_response" "integration_response_api_gw_valid_captcha" {
  depends_on = [
    aws_api_gateway_integration.integration_api_gw_valid_captcha
  ]
  rest_api_id = aws_api_gateway_rest_api.api_gw_inference_pipeline.id
  resource_id = aws_api_gateway_resource.resource_api_gw_valid_captcha.id
  http_method = aws_api_gateway_method.method_api_gw_valid_captcha.http_method
  status_code = aws_api_gateway_method_response.method_response_api_gw_valid_captcha.status_code

  response_templates = {
    "application/json" = "Empty"
  }
}

# Get Captcha
resource "aws_api_gateway_resource" "resource_api_gw_get_captcha" {
  depends_on = [
    aws_api_gateway_rest_api.api_gw_inference_pipeline
  ]
  rest_api_id = aws_api_gateway_rest_api.api_gw_inference_pipeline.id
  parent_id   = aws_api_gateway_rest_api.api_gw_inference_pipeline.root_resource_id
  path_part   = "get_captcha"
}

resource "aws_api_gateway_method" "method_api_gw_get_captcha" {
  depends_on = [
    aws_api_gateway_resource.resource_api_gw_get_captcha
  ]
  rest_api_id   = aws_api_gateway_rest_api.api_gw_inference_pipeline.id
  resource_id   = aws_api_gateway_resource.resource_api_gw_get_captcha.id
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_method_response" "method_response_api_gw_get_captcha" {
  depends_on = [
    aws_api_gateway_method.method_api_gw_get_captcha
  ]
  rest_api_id     = aws_api_gateway_rest_api.api_gw_inference_pipeline.id
  resource_id     = aws_api_gateway_resource.resource_api_gw_get_captcha.id
  http_method     = aws_api_gateway_method.method_api_gw_get_captcha.http_method
  status_code     = "200"
  response_models = {
    "application/json" = "Empty"
  }
}

resource "aws_api_gateway_integration" "integration_api_gw_get_captcha" {
  depends_on = [
    aws_api_gateway_method.method_api_gw_get_captcha
  ]
  rest_api_id             = aws_api_gateway_rest_api.api_gw_inference_pipeline.id
  resource_id             = aws_api_gateway_resource.resource_api_gw_get_captcha.id
  http_method             = aws_api_gateway_method.method_api_gw_get_captcha.http_method
  type                    = "AWS_PROXY"
  integration_http_method = "POST"
  uri                     = aws_lambda_function.lambda_function_inference_pipeline.invoke_arn
}

resource "aws_api_gateway_integration_response" "integration_response_api_gw_get_captcha" {
  depends_on = [
    aws_api_gateway_integration.integration_api_gw_get_captcha
  ]
  rest_api_id = aws_api_gateway_rest_api.api_gw_inference_pipeline.id
  resource_id = aws_api_gateway_resource.resource_api_gw_get_captcha.id
  http_method = aws_api_gateway_method.method_api_gw_get_captcha.http_method
  status_code = aws_api_gateway_method_response.method_response_api_gw_get_captcha.status_code

  response_templates = {
    "application/json" = "Empty"
  }
}

# Deployment
resource "aws_api_gateway_deployment" "deployment_api_gw_inference_pipeline" {
  depends_on = [
    aws_api_gateway_integration.integration_api_gw_inference_pipeline
  ]
  rest_api_id = aws_api_gateway_rest_api.api_gw_inference_pipeline.id
}

resource "aws_api_gateway_stage" "stage_api_gw_inference_pipeline" {
  depends_on = [
    aws_api_gateway_deployment.deployment_api_gw_inference_pipeline
  ]
  deployment_id = aws_api_gateway_deployment.deployment_api_gw_inference_pipeline.id
  rest_api_id   = aws_api_gateway_rest_api.api_gw_inference_pipeline.id
  stage_name    = var.api_name_inference_pipeline
}
