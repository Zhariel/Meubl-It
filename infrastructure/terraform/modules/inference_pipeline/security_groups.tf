################ Security Group ################

resource "aws_security_group" "sg_lambda_inference_pipeline" {
  vpc_id = aws_vpc.my_vpc.id

  ingress {
    protocol  = -1
    self      = true
    from_port = 0
    to_port   = 0
  }

  ingress {
    from_port = 0
    protocol  = "tcp"
    to_port   = 65535
    cidr_blocks = []
  }

  ingress {
    from_port = 80
    protocol  = "tcp"
    to_port   = 80
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port = 0
    protocol  = "tcp"
    to_port   = 65535
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "Lambda Inference Pipeline - Security Group"
  }
}