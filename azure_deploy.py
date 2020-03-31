import mlflow.azureml

from azureml.core import Workspace
from azureml.core.webservice import AciWebservice, Webservice


# Create or load an existing Azure ML workspace. You can also load an existing workspace using
# Workspace.get(name="<workspace_name>")
workspace_name = "example_mlflow"
subscription_id = "<>"
resource_group = "example_mlflow_resgrp"
location = "South Central US"

workspace = True
if workspace:
    azure_workspace = Workspace.get(name=workspace_name,
                                    subscription_id=subscription_id,
                                    resource_group=resource_group)
else:
    azure_workspace = Workspace.create(name=workspace_name,
                                   subscription_id=subscription_id,
                                   resource_group=resource_group,
                                   location=location,
                                   create_resource_group=False,
                                   )


print("Fetched workspace")
# Build an Azure ML container image for deployment
azure_image, azure_model = mlflow.azureml.build_image(model_uri="examplemodel",
                                                      workspace=azure_workspace,
                                                      description="examplewinesklearn",
                                                      synchronous=True)



# If your image build failed, you can access build logs at the following URI:
print("Access the following URI for build logs: {}".format(azure_image.image_build_log_uri))



import logging
logging.basicConfig(level=logging.DEBUG)
# Deploy the container image to ACI
webservice_deployment_config = AciWebservice.deploy_configuration()
print("reached_here")


webservice = Webservice.deploy_from_image(
                    image=azure_image, workspace=azure_workspace, deployment_config=webservice_deployment_config, name="sklearnexamplewine")
##It must only consist of lowercase letters, numbers, or dashes, start with a letter, end with a letter or number, and be between 3 and 32 characters long.
print(webservice.get_logs())
webservice.wait_for_deployment(show_output=True)

# After the image deployment completes, requests can be posted via HTTP to the new ACI
# webservice's scoring URI. The following example posts a sample input from the wine dataset
# used in the MLflow ElasticNet example:
# https://github.com/mlflow/mlflow/tree/master/examples/sklearn_elasticnet_wine
print("Scoring URI is: %s", webservice.scoring_uri)