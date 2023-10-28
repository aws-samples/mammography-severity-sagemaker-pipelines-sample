######################################################
# Create the SNS topic
######################################################

 resource "aws_sns_topic" "pipeline_sns_topic" {
   name = var.pipelines_alert_topic_name
   kms_master_key_id = "alias/aws/sns"
 }

######################################################
# Step 1: Create the SNS topic policy
######################################################

resource "aws_sns_topic_policy" "pipeline_sns_topic_policy" {
   arn = aws_sns_topic.pipeline_sns_topic.arn
   policy                                   = jsonencode(
         {
             Id        = "Pipeline-Status-Change-Topic-Policy"
             Statement = [
                 {
                     Action    = "sns:Publish"
                     Effect    = "Allow"
                     Principal = {
                         Service = "events.amazonaws.com"
                     }
                     Resource  = aws_sns_topic.pipeline_sns_topic.arn
                     Sid       = "SID-Pipeline-Status-Change"
                 },
             ]
             Version   = "2012-10-17"
         }
     )
 }
 
######################################################
# Step 2: Create the topic subscription 
######################################################
 resource "aws_sns_topic_subscription" "user_updates_sqs_target" {
   topic_arn              = aws_sns_topic.pipeline_sns_topic.arn
   protocol               = "email"
   endpoint               = var.email
   endpoint_auto_confirms = false
 }

