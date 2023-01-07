terraform {
  backend "s3" {
    bucket = "sethu-tf-state"
    key    = "tf_state"
    region = "us-east-1"
  }
}

provider "aws" {
  region = "us-east-1"
}

module "phising_feature_store" {
  source = "./phising_feature_store"
}

module "phising_app_artifacts" {
  source = "./phising_artifacts"
}

module "phising_model_ecr" {
  source = "./phising_model_ecr"
}

module "phising_app_runner" {
  source = "./phising_app_runner"
}

module "phising_mlflow_instance" {
  source = "./phising_mlflow_instance"
}