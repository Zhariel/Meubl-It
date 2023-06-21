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

# retrain Pipeline
variable "ecr_repository_name_retrain_pipeline" {
  default = ""
}

variable "iam_role_lambda_name_retrain_pipeline" {
  default = ""
}
variable "lambda_function_name_retrain_pipeline" {
  default = ""
}

variable "bucket_name_retrain_pipeline" {
  default = ""
}

variable "api_python_file_name_retrain_pipeline" {
  default = ""
}

variable "api_docker_file_name_retrain_pipeline" {
  default = ""
}

variable "lambda_function_memory_size_retrain_pipeline" {
  default = "512"
}

variable "lambda_function_timeout_retrain_pipeline" {
  default = "300"
}

variable "bucket_key_model_retrain_pipeline" {
  default = ""
}

variable "sg_lambda_retrain_pipeline_id" {
  default = ""
}

variable "private_subnet_retrain_pipeline_id" {
    default = ""
}

variable "s3_bucket_retrain_pipeline_id" {
  default = ""
}

variable "s3_object_annotated_retrain_pipeline_key" {
    default = ""
}
