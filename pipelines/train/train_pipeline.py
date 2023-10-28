import os
import time
import boto3
import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession

from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat,
)

from sagemaker.sklearn.processing import SKLearnProcessor

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep

from sagemaker.processing import ScriptProcessor
from sagemaker.model_monitor import DatasetFormat, model_monitoring
from sagemaker.drift_check_baselines import DriftCheckBaselines


from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.workflow.execution_variables import ExecutionVariables


from sagemaker import Model 
from sagemaker.workflow.model_step import ModelStep

from sagemaker.model_metrics import MetricsSource, ModelMetrics

from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join

from sagemaker.workflow.conditions import ConditionGreaterThan
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet

from sagemaker.workflow.pipeline import Pipeline


def get_pipeline(
    region=None,
    role=None,
    default_bucket=None,
    model_artifacts_bucket=None,
    kms_key=None,
    model_name = None,
    model_package_group_name="mammo-severity-package-group",
    pipeline_name="mammo-severity-train-pipeline",
    base_job_prefix="mammo-severity"
):
    """Gets a SageMaker ML Train Pipeline instance working with mammography dataset.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the data and predictions
        model_artifacts_bucket: the bucket in central model registry to store model artifacts
        model_name: model name used to create job names and prefixes in S3 bucket

    Returns:
        an instance of a pipeline
    """
    
    auc_score_threshold = ParameterFloat(name="AucScoreThreshold", default_value=0.5)
    
    # create pipeline session
    boto_session = boto3.session.Session(region_name=region)
    pipeline_session = PipelineSession(boto_session=boto_session)
    
    # set the S3 bucket prefix based on timestamp
    bucket_prefix = f"{model_name}/{int(time.time())}"
    
    # Parameters for Pipeline Execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1) 
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.xlarge"
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=f"s3://{default_bucket}/{model_name}/data/train-dataset/",
    )
    
    # STEP 1: Processing Step for Feature Engineering
    sklearn_processor = SKLearnProcessor(
        framework_version="1.0-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        output_kms_key =kms_key,
        base_job_name="mammo-process-step",
        role=role,
        sagemaker_session=pipeline_session,
    )
    processor_args = sklearn_processor.run(
            inputs=[
              ProcessingInput(source=input_data, destination="/opt/ml/processing/input"),  
            ],
            outputs=[
                ProcessingOutput(output_name="train", source="/opt/ml/processing/train",\
                                 destination=f"s3://{default_bucket}/{bucket_prefix}/output/train" ),
                ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation",\
                                destination=f"s3://{default_bucket}/{bucket_prefix}/output/validation"),
                ProcessingOutput(output_name="test", source="/opt/ml/processing/test",\
                                destination=f"s3://{default_bucket}/{bucket_prefix}/output/test"),
                ProcessingOutput(output_name="baseline", source="/opt/ml/processing/baseline",\
                                destination=f"s3://{default_bucket}/{bucket_prefix}/output/baseline"),
            ],
            code=f"s3://{default_bucket}/{model_name}/scripts/raw_preprocess.py",
    )
    step_process = ProcessingStep(name="Mammo-PreProcessing", step_args=processor_args)
    
    # STEP 2: Training Step
    model_path = f"s3://{model_artifacts_bucket}/{bucket_prefix}/model"
    image_uri = sagemaker.image_uris.retrieve( 
        framework="xgboost",
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type="ml.m5.xlarge",
    )
    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        output_path=model_path,
        base_job_name=f"mammo-train-step",
        role=role,
        output_kms_key=kms_key,
        sagemaker_session=pipeline_session,
    )
    xgb_train.set_hyperparameters( 
        objective="binary:logistic", 
        eval_metric="auc",
        num_round=50,
        max_depth=5,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.7,
    )
    
    train_args = xgb_train.fit( 
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        }
    )
    
    step_train = TrainingStep(
        name="Mammo-Train",
        step_args=train_args,
    )
    
    # STEP 3: Model Evaluation Step
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name="script-mammo-eval",
        role=role,
        output_kms_key=kms_key,
        sagemaker_session=pipeline_session,
    )
    
    eval_args = script_eval.run(
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation",\
                                 destination=f"s3://{default_bucket}/{bucket_prefix}/output/evaluation"),

        ],
        code=f"s3://{default_bucket}/{model_name}/scripts/evaluate_model.py",
    )
    
    evaluation_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation", path="evaluation.json"
    )
    
    step_eval = ProcessingStep(
        name="Mammo-Eval",
        step_args=eval_args,
        property_files=[evaluation_report],
    )
    
    # STEP 4: Configure the Data Quality Baseline Job

    # Configure the transient compute environment
    check_job_config = CheckJobConfig(
        role=role,
        instance_count=1,
        instance_type="ml.c5.xlarge",
        volume_size_in_gb=120,
        output_kms_key=kms_key,
        sagemaker_session=pipeline_session,
    )

    # Configure the data quality check input (training data), dataset format, and S3 output path
    data_quality_check_config = DataQualityCheckConfig(
        baseline_dataset=step_process.properties.ProcessingOutputConfig.Outputs['baseline'].S3Output.S3Uri,
        dataset_format=DatasetFormat.csv(header=False, output_columns_position="START"),
        output_s3_uri=Join(on='/', values=['s3:/', model_artifacts_bucket, bucket_prefix, ExecutionVariables.PIPELINE_EXECUTION_ID, 'dataqualitycheckstep'])
    )
    
    # Configure Pipeline Step - 'QualityCheckStep'
    baseline_model_data_step = QualityCheckStep(
            name="Mammo-DataQualityCheck",
            # skip_check, indicates a baselining job
            skip_check=True,
            register_new_baseline=True,
            quality_check_config=data_quality_check_config,
            check_job_config=check_job_config,
            model_package_group_name=model_package_group_name
        )
    
    
    # STEP 5: Register Model in Model Registry
    
    # Specify model metric and drift baseline metadata to register
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri='{}/evaluation.json'.format(
                step_eval.arguments['ProcessingOutputConfig']['Outputs'][0]['S3Output']['S3Uri']
            ),
            content_type='application/json',
        ),
        model_data_statistics=MetricsSource(
                s3_uri=baseline_model_data_step.properties.CalculatedBaselineStatistics,
                content_type="application/json",
        ),
        model_data_constraints=MetricsSource(
                s3_uri=baseline_model_data_step.properties.CalculatedBaselineConstraints,
                content_type="application/json",
        ),
    )

    drift_check_baselines = DriftCheckBaselines(
        model_data_statistics=MetricsSource(
            s3_uri=baseline_model_data_step.properties.BaselineUsedForDriftCheckStatistics,
            content_type="application/json",
        ),
        model_data_constraints=MetricsSource(
            s3_uri=baseline_model_data_step.properties.BaselineUsedForDriftCheckConstraints,
            content_type="application/json",
        ),
    )

    model = Model(
        image_uri=image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=role,
    )


    # Configure step to register model version using metadata and Model object: model.register()
    model_registry_args = model.register(
        content_types=['text/csv'],
        response_types=['text/csv'],
        inference_instances=['ml.t2.medium', 'ml.m5.xlarge'],
        transform_instances=['ml.m5.xlarge'],
        model_package_group_name=model_package_group_name,
        drift_check_baselines=drift_check_baselines,
        approval_status=model_approval_status,
        model_metrics=model_metrics
    )
    
    step_register = ModelStep(name="Mammo-RegisterModel", step_args=model_registry_args)
    
    # STEP 5: Condition Step to check AUC Score
    cond_lte = ConditionGreaterThan(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="classification_metrics.auc.value",
        ),
        right=auc_score_threshold,
    )
    
    step_fail = FailStep(
        name="Mammo-AUCFail",
        error_message=Join(on=" ", values=["Execution failed due to Model's AUC <", auc_score_threshold]),
    )
    
    step_cond = ConditionStep(
        name="Mammo-CheckAUCScore",
        conditions=[cond_lte],
        if_steps=[baseline_model_data_step,step_register],
        else_steps=[step_fail],
    )
    
    # Pipeline Instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            model_approval_status,
            input_data,
            auc_score_threshold,
        ],
        steps=[step_process, step_train,step_eval, step_cond],
        sagemaker_session=pipeline_session
    )
    
    return pipeline
