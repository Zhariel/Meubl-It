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

variable "s3_function_name_inference_pipeline" {
  default = ""
}

variable "iam_role_s3_name_inference_pipeline" {
  default = ""
}
variable local_path {
  default = ""
}
