import boto3
import sagemaker 
import time
import os

from sagemaker.transformer import Transformer
from sagemaker.inputs import TransformInput
from sagemaker.workflow.steps import TransformStep
from sagemaker.sklearn.processing import SKLearnProcessor

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.monitor_batch_transform_step import MonitorBatchTransformStep
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat,
)
from sagemaker.workflow.lambda_step import (
    LambdaStep,
    LambdaOutput,
    LambdaOutputTypeEnum,
)
from sagemaker.lambda_helper import Lambda
from sagemaker import Model 
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.quality_check_step import DataQualityCheckConfig
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.workflow.pipeline import Pipeline


def get_pipeline(
    region=None,
    role=None,
    lambda_role=None,
    default_bucket=None,
    kms_key=None,
    model_name = None,
    model_package_group_name="mammo-severity-package-group",
    pipeline_name ="mammo-severity-inference-pipeline",
    batch_dataset_filename = "mammo-batch-dataset"

):
    """Gets a SageMaker ML Inference Pipeline instance working with mammography dataset.
    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the data and predictions
        model_name: model name used to create job names and prefixes in S3 bucket
    Returns:
        an instance of a pipeline
    """

    # create pipeline session
    boto_session = boto3.session.Session(region_name=region)
    pipeline_session = PipelineSession(boto_session=boto_session)

    # set the S3 bucket prefix based on timestamp
    bucket_prefix = f"{model_name}/{int(time.time())}"
    function_name = "sagemaker-lambda-step-latest-approved-model"

    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.xlarge"
    )
    
    transform_input_param = ParameterString(
        name="transform_input",
        default_value=f"s3://{default_bucket}/{model_name}/data/batch-dataset/{batch_dataset_filename}.csv",
    )
    batch_monitor_reports_output_path = "s3://{}/{}/data-quality-monitor-reports".format(default_bucket, bucket_prefix)
    
    # Step 1: Get Latest Approved Model with Lambda Step
    
    func = Lambda(
        function_name=function_name,
        execution_role_arn=lambda_role,
        session=pipeline_session,
        s3_bucket = default_bucket,
        script=f"pipelines/inference/scripts/lambda_helper.py",
        handler="lambda_helper.lambda_handler",
        )
    
    model_artifact_uri = LambdaOutput(output_name="model_artifact_uri", output_type=LambdaOutputTypeEnum.String)
    image_uri = LambdaOutput(output_name="image_uri", output_type=LambdaOutputTypeEnum.String)
    s3_baseline_uri_statistics = LambdaOutput(output_name="s3_baseline_uri_statistics", output_type=LambdaOutputTypeEnum.String)
    s3_baseline_uri_constraints = LambdaOutput(output_name="s3_baseline_uri_constraints", output_type=LambdaOutputTypeEnum.String)

    step_approved_model = LambdaStep(
        name="LambdaStep",
        lambda_func=func,
        inputs={
            "region": region,
            "model_package_group_name":model_package_group_name,
        },
        outputs=[model_artifact_uri, image_uri, s3_baseline_uri_statistics,s3_baseline_uri_constraints],
    )

    # Step 2: Create Model Step
    model = Model(
        image_uri=step_approved_model.properties.Outputs["image_uri"],
        model_data=step_approved_model.properties.Outputs["model_artifact_uri"],
        sagemaker_session=pipeline_session,
        role=role
        )

    step_create_model = ModelStep(
        name="Mammo-CreateModel",
        step_args=model.create(instance_type="ml.m5.large", accelerator_type="ml.eia1.medium"),
        )

    # Step 3: Define processing step
    sklearn_processor = SKLearnProcessor(
        framework_version="1.0-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name="mammo-process-step",
        output_kms_key=kms_key,
        role=role,
        sagemaker_session=pipeline_session,
    )
    processor_args = sklearn_processor.run(
            inputs=[
              ProcessingInput(source=transform_input_param, destination="/opt/ml/processing/input"),  
            ],
            outputs=[
                ProcessingOutput(output_name="inference", source="/opt/ml/processing/inference",\
                                 destination=f"s3://{default_bucket}/{bucket_prefix}/output/inference" ),
            ],
            code=f"s3://{default_bucket}/{model_name}/scripts/batch_preprocess.py",
    )
    step_process = ProcessingStep(name="Mammo-PreProcessing", step_args=processor_args)

    # Step 4: Define Batch Transform Step
    transformer = Transformer(
        model_name=step_create_model.properties.ModelName,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        output_kms_key=kms_key,
        output_path=f"s3://{default_bucket}/{bucket_prefix}/output/inference",
        sagemaker_session=pipeline_session
    )
    
    transform_arg = transformer.transform(
        data = transform_input_param,
        content_type="text/csv",
        split_type="Line",
        )
    
    job_config = CheckJobConfig(role=role,sagemaker_session=pipeline_session)
    data_quality_config = DataQualityCheckConfig(
        baseline_dataset=transform_input_param,
        dataset_format=DatasetFormat.csv(header=False),
        output_s3_uri=batch_monitor_reports_output_path,
    )
    
    transform_and_monitor_step = MonitorBatchTransformStep(
        name="MonitorCustomerChurnDataQuality",
        transform_step_args=transform_arg,
        monitor_configuration=data_quality_config,
        check_job_configuration=job_config,
        monitor_before_transform=True,
        fail_on_violation=True,
        supplied_baseline_statistics=step_approved_model.properties.Outputs["s3_baseline_uri_statistics"],
        supplied_baseline_constraints=step_approved_model.properties.Outputs["s3_baseline_uri_constraints"],

        )   

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            transform_input_param
        ],
        steps=[step_approved_model, step_create_model, transform_and_monitor_step],
        sagemaker_session=pipeline_session,
    )
    return pipeline