
"""
This Lambda function gets information of the latest approved model from the central model registry
"""

import boto3
import json

def lambda_handler(event, context):
    
    sm_client = boto3.client('sagemaker', region_name=event['region'])

    # get a list of approved model packages from the model package group specified 
    approved_model_packages = sm_client.list_model_packages(
          ModelApprovalStatus='Approved',
          ModelPackageGroupName=event["model_package_group_name"],
          SortBy='CreationTime',
          SortOrder='Descending'
      )

    # find the latest approved model package
    try:
        latest_approved_model_package_arn = approved_model_packages['ModelPackageSummaryList'][0]['ModelPackageArn']
    except Exception as e:
        print("Failed to retrieve an approved model package:", e)

    # retrieve required information about the model
    latest_approved_model_package_descr =  sm_client.describe_model_package(ModelPackageName = latest_approved_model_package_arn)

    # model artifact uri (tar.gz file)
    model_artifact_uri = latest_approved_model_package_descr['InferenceSpecification']['Containers'][0]['ModelDataUrl']
    # sagemaker image in ecr
    image_uri = latest_approved_model_package_descr['InferenceSpecification']['Containers'][0]['Image']

    # get baseline metrics
    s3_baseline_uri_statistics = latest_approved_model_package_descr["ModelMetrics"]["ModelDataQuality"]["Statistics"]["S3Uri"]
    s3_baseline_uri_constraints = latest_approved_model_package_descr["ModelMetrics"]["ModelDataQuality"]["Constraints"]["S3Uri"]

    return {
        "model_artifact_uri": model_artifact_uri,
        "image_uri": image_uri,
        "s3_baseline_uri_statistics": s3_baseline_uri_statistics,
        "s3_baseline_uri_constraints": s3_baseline_uri_constraints
    }
