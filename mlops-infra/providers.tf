provider "aws" {
  region = var.aws_region
  default_tags {
    tags = {
      "environment" = var.env_group
      "team"        = "MLOpsAdmin"
    }
  }
}