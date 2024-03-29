output "sg_lambda_inference_pipeline_id" {
  value = aws_security_group.sg_lambda_inference_pipeline.id
}

output "private_subnet_inference_pipeline_id" {
  value = aws_subnet.private_subnet_inference_pipeline.id
}

output "s3_bucket_inference_pipeline_id" {
  value = aws_s3_bucket.s3_meubl_it_bucket.id
}

output "s3_object_annotated_inference_pipeline_key" {
  value = aws_s3_object.s3_object_annotated.key
}

output "iam_policy_process_logging_policy_arn" {
  value = aws_iam_policy.process_logging_policy.arn
}
