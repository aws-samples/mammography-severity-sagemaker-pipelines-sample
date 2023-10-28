module "pipeline_sns_topic" {
  source = "./modules/sns-notifications"
  email  = var.email
  pipelines_alert_topic_name = var.pipelines_alert_topic_name
}

module "mlops-event-bridge-triggers" {
  source  = "./modules/event_bridge"  
  environment = var.env_group
  train_pipeline = var.train_pipeline
  inference_pipeline = var.inference_pipeline
  event_bus_name = var.event_bus_name
  pipelines_alert_topic_name = var.pipelines_alert_topic_name

}
