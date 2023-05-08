################ Virtual Private Cloud ################

resource "aws_vpc" "my_vpc" {
  cidr_block = "10.0.0.0/16"
  enable_dns_hostnames = true
  tags = {
    Name = "My VPC"
  }
}

data "aws_availability_zones" "availability_zones" {
  state = "available"
}



resource "aws_vpc_endpoint" "s3" {
  vpc_endpoint_type   = "Interface"
  vpc_id              = aws_vpc.my_vpc.id
  service_name        = "com.amazonaws.${var.region}.s3"
  security_group_ids  = [aws_security_group.sg_lambda_inference_pipeline.id]
  subnet_ids          = [aws_subnet.private_subnet.id]
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

resource "aws_subnet" "private_subnet" {
  vpc_id = aws_vpc.my_vpc.id
  cidr_block = "10.0.2.0/24"
  availability_zone = data.aws_availability_zones.availability_zones.names[0]
  tags = {
    Name = "Private Subnet - ${data.aws_availability_zones.availability_zones.names[0]}"
  }
}