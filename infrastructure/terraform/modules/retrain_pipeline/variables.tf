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
variable "s3_meubl_it_bucket_name" {
  default = ""
}

variable "s3_meubl_it_bucket_id" {
  default = ""
}

variable "s3_object_meubl_it_model_key" {
  default = ""
}

variable "s3_object_annotated_data_key" {
  default = ""
}

variable "iam_policy_process_logging_policy_arn" {
  default = ""
}

variable "sg_lambda_inference_pipeline_id" {
  default = ""
}

variable "private_subnet_inference_pipeline_id" {
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
