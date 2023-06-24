# General
variable "region" {
  default = ""
}

variable "profile_name" {
  default = ""
}

variable "os" {
  default = ""
}

# Inference Pipeline
variable "ecr_repository_name_inference_pipeline" {
  default = ""
}

variable "iam_role_lambda_name_inference_pipeline" {
  default = ""
}

variable "lambda_function_name_inference_pipeline" {
  default = ""
}

variable "s3_meubl_it_bucket_name" {
  default = ""
}

variable "api_name_inference_pipeline" {
  default = ""
}

variable "api_description_inference_pipeline" {
  default = ""
}

variable "python_file_path_api_inference_pipeline" {
  default = ""
}

variable "docker_file_path_api_inference_pipeline" {
  default = ""
}

variable "lambda_function_memory_size_inference_pipeline" {
  default = 512
}

variable "lambda_function_timeout_inference_pipeline" {
  default = 300
}

variable "s3_object_meubl_it_model_key" {
  default = ""
}

variable "meubl_it_model_path" {
  default = ""
}

# Retrain Pipeline
variable "ecr_repository_name_retrain_pipeline" {
  default = ""
}

variable "iam_role_lambda_name_retrain_pipeline" {
  default = ""
}

variable "lambda_function_name_retrain_pipeline" {
  default = ""
}

variable "python_file_path_retrain_pipeline" {
  default = ""
}

variable "docker_file_path_retrain_pipeline" {
  default = ""
}

variable "lambda_function_memory_size_retrain_pipeline" {
  default = 512
}

variable "lambda_function_timeout_retrain_pipeline" {
  default = 300
}
