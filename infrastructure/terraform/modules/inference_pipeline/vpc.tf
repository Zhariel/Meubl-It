################ Virtual Private Cloud ################

resource "aws_vpc" "vpc_inference_pipeline" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = {
    Name = "My VPC"
  }
}

data "aws_availability_zones" "availability_zones" {
  state = "available"
}

#resource "aws_subnet" "public" {
#  count = 2
#  vpc_id = aws_vpc.my_vpc.id
#  cidr_block = "10.0.1.0/24"
#  availability_zone = element(data.aws_availability_zones.availability_zones.names, count.index)
#  tags = {
#    Name = "Public Subnet - ${element(data.aws_availability_zones.availability_zones.names, count.index)}"
#  }
#}

#resource "aws_internet_gateway" "my_vpc_igw" {
#  vpc_id = aws_vpc.my_vpc.id
#  tags = {
#    Name = "My VPC - Internet Gateway"
#  }
#}

resource "aws_subnet" "private_subnet_inference_pipeline" {
  vpc_id                  = aws_vpc.vpc_inference_pipeline.id
  cidr_block              = "10.0.2.0/24"
  map_public_ip_on_launch = false
  availability_zone       = data.aws_availability_zones.availability_zones.names[0]
  tags                    = {
    Name = "Private Subnet - ${data.aws_availability_zones.availability_zones.names[0]}"
  }
}

resource "aws_vpc_endpoint" "vpc_endpoint_s3_inference_pipeline" {
  vpc_id          = aws_vpc.vpc_inference_pipeline.id
  service_name    = "com.amazonaws.${var.region}.s3"
  route_table_ids = [aws_route_table.private_route_table_inference_pipeline.id]
}

