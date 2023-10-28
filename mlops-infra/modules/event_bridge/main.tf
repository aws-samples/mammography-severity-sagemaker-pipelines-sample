data "aws_region" "current" {}
data "aws_caller_identity" "current" {}

###########################################################
# SNS Topic for SageMaker Pipeline Failure Notification #
###########################################################

resource "aws_sns_topic" "sagmaker_failure_alert_notification" {
  name = var.pipelines_alert_topic_name
  kms_master_key_id = "alias/aws/sns"
} 

##################################################
# EventBridge rule to schedule inference pipeline #
##################################################

resource "aws_cloudwatch_event_rule" "inference_pipeline_rule" {
  
  name        = "${var.inference_pipeline["name"]}-rule"
  description = "Rule for triggering ${var.inference_pipeline["name"]} on schedule"
  schedule_expression = var.inference_pipeline["cron_schedule"]
  event_bus_name = var.event_bus_name
  role_arn = var.inference_pipeline["role_arn"]
}

resource "aws_cloudwatch_event_target" "inference_pipeline_target" {
  
  target_id = "${var.inference_pipeline["name"]}-target"
  rule      = aws_cloudwatch_event_rule.inference_pipeline_rule.name
  arn       = var.inference_pipeline["arn"]
  event_bus_name = var.event_bus_name
  role_arn = var.inference_pipeline["role_arn"]
 
}

######################################################
# EventBridge rule to trigger train pipeline and send
# out notification on inference pipeline failure 
######################################################

resource "aws_cloudwatch_event_rule" "inference_pipeline_failure_alert" {
    
  name        = "${var.inference_pipeline["name"]}-failure"
  description = "Rule for triggering alert on ${var.inference_pipeline["name"]} failure"
  event_pattern ="{\"source\":[\"aws.sagemaker\"],\"detail-type\":[\"SageMaker Model Building Pipeline Execution Status Change\"],\"detail\":{\"currentPipelineExecutionStatus\":[\"Failed\"],\"pipelineArn\":[\"${var.inference_pipeline["arn"]}\"]}}"
   }


resource "aws_cloudwatch_event_target" "inference_pipeline_failure_alert_target_sns" {
  
  target_id = "${var.inference_pipeline["name"]}-failure-sns"
  rule      = aws_cloudwatch_event_rule.inference_pipeline_failure_alert.name
  arn       = aws_sns_topic.sagmaker_failure_alert_notification.arn
  event_bus_name = var.event_bus_name
 
}

resource "aws_cloudwatch_event_target" "inference_pipeline_failure_alert_target_smpipeline" {
  
  target_id = "${var.inference_pipeline["name"]}-failure-smpipeline"
  rule      = aws_cloudwatch_event_rule.inference_pipeline_failure_alert.name
  arn       = var.train_pipeline["arn"]
  event_bus_name = var.event_bus_name
  role_arn = var.train_pipeline["role_arn"]

 
}

##################################################
# EventBridge rule to send out notification
# on train pipeline failure 
##################################################

resource "aws_cloudwatch_event_rule" "train_pipeline_failure_alert" {
    
  name        = "${var.train_pipeline["name"]}-failure"
  description = "Rule for triggering alert on ${var.train_pipeline["name"]} failure"
  event_pattern ="{\"source\":[\"aws.sagemaker\"],\"detail-type\":[\"SageMaker Model Building Pipeline Execution Status Change\"],\"detail\":{\"currentPipelineExecutionStatus\":[\"Failed\"],\"pipelineArn\":[\"${var.train_pipeline["arn"]}\"]}}"

}


resource "aws_cloudwatch_event_target" "train_pipeline_failure_alert_target_sns" {
  
  target_id = "${var.train_pipeline["name"]}-failure-sns"
  rule      = aws_cloudwatch_event_rule.train_pipeline_failure_alert.name
  arn       = aws_sns_topic.sagmaker_failure_alert_notification.arn
  event_bus_name = var.event_bus_name
 
}

