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


variable "bucket_name_inference_pipeline" {
  default = ""
}

variable "bucket_name_inference_pipeline_data" {
  default = ""
}

variable "api_name_inference_pipeline" {
  default = ""
}

variable "api_description_inference_pipeline" {
  default = ""
}

variable "api_python_file_name_inference_pipeline" {
  default = ""
}

variable "api_docker_file_name_inference_pipeline" {
  default = ""
}

variable "lambda_function_memory_size_inference_pipeline" {
  default = "512"
}

variable "lambda_function_timeout_inference_pipeline" {
  default = "300"
}

variable "bucket_key_model_inference_pipeline" {
  default = ""
}

variable "model_path_inference_pipeline" {
  default = ""
}

# retrain pipeline
variable "lambda_function_name_retrain_pipeline" {
  default = ""
}