################ Elastic Container Registry ################

locals {
  ecr_image_tag = "latest"
}

data aws_caller_identity current {}

data aws_ecr_authorization_token token {}

resource "aws_ecr_repository" "ecr_repo_retrain_pipeline" {
  name         = var.ecr_repository_name_retrain_pipeline
  force_delete = true

}

resource null_resource tensorflow_image {
  depends_on = [aws_ecr_repository.ecr_repo_retrain_pipeline]
  triggers   = {
    python_file = sha256(file(var.python_file_path_retrain_pipeline))
    docker_file = sha256(file(var.docker_file_path_retrain_pipeline))
  }
  provisioner "local-exec" {
    command     = <<EOF
             cd ../../retrain-pipeline
             docker logout ${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.region}.amazonaws.com
             docker login --username ${data.aws_ecr_authorization_token.token.user_name} --password ${data.aws_ecr_authorization_token.token.password} ${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.region}.amazonaws.com
             docker build -t ${aws_ecr_repository.ecr_repo_retrain_pipeline.repository_url}:${local.ecr_image_tag} .
             docker push ${aws_ecr_repository.ecr_repo_retrain_pipeline.repository_url}:${local.ecr_image_tag}
         EOF
    interpreter = var.os == "win" ? ["PowerShell", "-Command"] : []
  }
}

data aws_ecr_image ecr_tensorflow_image {
  depends_on = [
    null_resource.tensorflow_image
  ]
  repository_name = var.ecr_repository_name_retrain_pipeline
  image_tag       = local.ecr_image_tag
}
