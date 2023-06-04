################ Routes Tables ################

locals {
  private_rt_tag_name_inference_pipeline = "Private Route Table - ${aws_subnet.private_subnet_inference_pipeline.tags.Name}"
}

#resource "aws_route_table" "public_route" {
#  vpc_id = aws_vpc.my_vpc.id
#  route {
#    cidr_block = "0.0.0.0/0"
#    gateway_id = aws_internet_gateway.my_vpc_igw.id
#  }
#  tags = {
#    Name = "Public Subnet Route Table"
#  }
#}
#
#resource "aws_route_table_association" "public_route_table_association" {
#  count = 2
#  subnet_id = element(aws_subnet.public.*.id, count.index)
#  route_table_id = aws_route_table.public_route.id
#}

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
