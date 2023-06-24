################ Routes Tables ################

locals {
  private_rt_tag_name_inference_pipeline = "Private Route Table - ${aws_subnet.private_subnet_inference_pipeline.tags.Name}"
}

resource "aws_route_table" "private_route_table_inference_pipeline" {
  vpc_id = aws_vpc.vpc_inference_pipeline.id
  tags   = {
    Name = local.private_rt_tag_name_inference_pipeline
  }
}

resource "aws_route_table_association" "private_route_table_association_inference_pipeline" {
  subnet_id      = aws_subnet.private_subnet_inference_pipeline.id
  route_table_id = aws_route_table.private_route_table_inference_pipeline.id
}
