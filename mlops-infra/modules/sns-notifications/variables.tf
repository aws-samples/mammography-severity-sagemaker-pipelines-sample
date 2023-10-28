variable "pipelines_alert_topic_name" {
  description = "Name of the SNS Topic to be created"
  default     = "Pipeline-Status-Notification"
}

variable "email" {
 description = "Email address for SNS"
 default = "admin@org.com"
}

