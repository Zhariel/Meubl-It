FROM public.ecr.aws/lambda/python:3.8
WORKDIR ${LAMBDA_TASK_ROOT}

RUN yum update -y && yum install mesa-libGL -y

COPY src .
COPY requirements.txt requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

CMD ["main.handler"]