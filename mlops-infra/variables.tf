variable env_group {
  type        = string
  description = "Enviromment Type"
  default = "prod"

}

variable train_pipeline {
  type = map
  default     = {}
  description = "Train Sagemaker Pipeline configs"
}

variable inference_pipeline {
  type = map
  default     = {}
  description = "Inference Sagemaker Pipeline configs"
}

variable event_bus_name {
  type        = string
  default     = "default"
  description = "Event Bus Name"
}

variable pipelines_alert_topic_name {
  type        = string
  description = "Pipelines SNS Topic Name"
} 

variable aws_region {
  type        = string
  description = "Aws Region"
  default = "us-east-1"
}

variable email {
  type        = string
  description = "email for notification"
  default = "admin@org.com"
}