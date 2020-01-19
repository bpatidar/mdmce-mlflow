import mlflow.sagemaker as mfs

account_id = "XXXXXXXXXXXX"
app_name = "textcateg"
model_uri = "mck_mlflow_pyfunc_linear_1"
role_arn = "arn:aws:iam::"+account_id+":role/service-role/AmazonSageMaker-ExecutionRole-20181123T171117"
bucket = "sagemaker-us-east-2-"+ account_id
image_url = account_id + ".dkr.ecr.us-east-2.amazonaws.com/mckesson"
region = "us-east-2"
mode = mfs.DEPLOYMENT_MODE_CREATE
instance_type = "ml.t2.large"
instance_count = 1
flavor = "python_function"
vpc_config = {"SecurityGroupIds": ["sg-0b4cc067"], "Subnets": ["subnet-583ff014"]}

##mlflow sagemaker deploy -a "textcateg" -m "mck_mlflow_pyfunc_linear_1" -e "arn:aws:iam::604136135526:role/service-role/AmazonSageMaker-ExecutionRole-20181123T171117" -b "sagemaker-us-east-2-604136135526" -i "604136135526.dkr.ecr.us-east-2.amazonaws.com/mckesson" --region-name "us-east-2" --mode "create" -t "ml.t2.medium" -c 1 -f "python_function" -v "vpc_config"
mfs.deploy(app_name=app_name,
           model_uri=model_uri,
           execution_role_arn=role_arn,
           bucket=bucket,
           image_url=image_url,
           region_name=region,
           mode=mode,
           archive=False,
           instance_type = instance_type,
           instance_count = instance_count,
           vpc_config=vpc_config,
           flavor=flavor,
           synchronous=True
           )

print("Deploy completed successfully")

import mlflow.sagemaker as awssmkr
##awssmkr.delete(app_name="textcateg",region_name='us-east-2', archive=False, synchronous=True)