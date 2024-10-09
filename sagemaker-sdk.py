from sagemaker.estimator import Estimator

role = "arn:aws:iam::015059194123:role/service-role/AmazonSageMaker-ExecutionRole-20241002T202930"
output_path = "s3://day-trade-predict/data/"

estimator = Estimator(
    image_uri="015059194123.dkr.ecr.us-east-1.amazonaws.com/private/day-trade-predict:latest",
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=output_path,
)

estimator.fit()
