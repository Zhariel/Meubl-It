module "inference_pipeline" {
  source                                         = "./modules/inference_pipeline"
  region                                         = var.region
  profile_name                                   = var.profile_name
  os                                             = var.os
  ecr_repository_name_inference_pipeline         = var.ecr_repository_name_inference_pipeline
  iam_role_lambda_name_inference_pipeline        = var.iam_role_lambda_name_inference_pipeline
  lambda_function_name_inference_pipeline        = var.lambda_function_name_inference_pipeline
  bucket_name_inference_pipeline                 = var.bucket_name_inference_pipeline
  api_name_inference_pipeline                    = var.api_name_inference_pipeline
  api_description_inference_pipeline             = var.api_description_inference_pipeline
  api_python_file_name_inference_pipeline        = var.api_python_file_name_inference_pipeline
  api_docker_file_name_inference_pipeline        = var.api_docker_file_name_inference_pipeline
  lambda_function_memory_size_inference_pipeline = var.lambda_function_memory_size_inference_pipeline
  lambda_function_timeout_inference_pipeline     = var.lambda_function_timeout_inference_pipeline
  bucket_key_model_inference_pipeline            = var.bucket_key_model_inference_pipeline
  model_path_inference_pipeline                  = var.model_path_inference_pipeline
}

module "retrain_pipeline" {
  source                                        = "./modules/retrain_pipeline"
  region                                        = var.region
  profile_name                                  = var.profile_name
  os                                            = var.os
  ecr_repository_name_inference_pipeline         = var.ecr_repository_name_inference_pipeline
  iam_role_lambda_name_inference_pipeline        = var.iam_role_lambda_name_inference_pipeline
  lambda_function_name_inference_pipeline        = var.lambda_function_name_inference_pipeline
  bucket_name_inference_pipeline                 = var.bucket_name_inference_pipeline
  api_python_file_name_inference_pipeline        = var.api_python_file_name_inference_pipeline
  api_docker_file_name_inference_pipeline        = var.api_docker_file_name_inference_pipeline
  lambda_function_memory_size_inference_pipeline = var.lambda_function_memory_size_inference_pipeline
  lambda_function_timeout_inference_pipeline     = var.lambda_function_timeout_inference_pipeline
  bucket_key_model_inference_pipeline            = var.bucket_key_model_inference_pipeline
  sg_lambda_retrain_pipeline_id = module.inference_pipeline.sg_lambda_inference_pipeline_id
  private_subnet_retrain_pipeline_id = module.inference_pipeline.private_subnet_inference_pipeline_id
  s3_object_annotated_retrain_pipeline_key = module.inference_pipeline.s3_object_annotated_inference_pipeline_key
  s3_object_trained_retrain_pipeline_key = module.inference_pipeline.s3_object_trained_inference_pipeline_key

}
