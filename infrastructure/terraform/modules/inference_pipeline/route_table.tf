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