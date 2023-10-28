variable "environment" {
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

variable "event_bus_name" {
  type        = string
  default     = "default"
  description = "Event Bus Name"
}

variable "pipelines_alert_topic_name" {
  type        = string
  default     = "default"
  description = "Pipelines SNS Topic Name"
}



