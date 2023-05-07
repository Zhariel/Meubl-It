

module "inference_pipeline" {
  source = "./modules/inference_pipeline"
  region = var.region
  profile_name = var.profile_name
  os = var.os
  ecr_repository_name_inference_pipeline = var.ecr_repository_name_inference_pipeline
  iam_role_lambda_name_inference_pipeline = var.iam_role_lambda_name_inference_pipeline
  lambda_function_name_inference_pipeline = var.lambda_function_name_inference_pipeline
  s3_function_name_inference_pipeline = var.s3_function_name_inference_pipeline
}