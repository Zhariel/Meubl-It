data aws_caller_identity current {}

data aws_ecr_authorization_token token {}

#terraform {
#  required_providers {
#    docker = {
#      source = "kreuzwerker/docker"
#      version = "~> 2.0"
#    }
#  }
#  required_version = ">= 0.13"
#}
#
#provider "docker" {
#  registry_auth {
#    address = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.region}.amazonaws.com"
#    username = data.aws_ecr_authorization_token.token.user_name
#    password = data.aws_ecr_authorization_token.token.password
#  }
#}
