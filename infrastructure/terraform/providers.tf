terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
      version = "~> 4.0"
    }
    null = {
      source  = "hashicorp/null"
    }
  }
  required_version = ">= 0.13"
}

provider "aws" {
  region  = var.region
  profile = var.profile_name
}