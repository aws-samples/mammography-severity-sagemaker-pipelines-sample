pipeline {
    agent any
     environment {
        PYTHONUNBUFFERED = 'TRUE'
        MODEL_ID = "mammography-severity-modeling"
        ENV = "prod" 
        }

    stages {
        stage('Read Config File, Build and Install') {
            agent any
            steps {  
                script {
                    def variables = readJSON file: "environments/${ENV}.tfvars.json"
                    env.REGION = "${variables.aws_region}"
                    env.SAGEMAKER_PIPELINE_ROLE = "${variables.train_pipeline["role_arn"]}"
                    env.LAMBDA_ROLE = "${variables.lambda_role}"
                    env.DEFAULT_BUCKET = "${variables.default_bucket}"
                    env.MODEL_ARTIFACTS_BUCKET = "${variables.model_artifacts_bucket}"
                    env.MODEL_NAME = "${variables.model_name}"
                    env.KMS_KEY = "${variables.kms_key}"
                    env.MODEL_PACKAGE_GROUP_NAME = "${variables.model_package_group_name}"
                    env.TRAIN_PIPELINE_NAME = "${variables.train_pipeline_name}"
                    env.INFERENCE_PIPELINE_NAME = "${variables.inference_pipeline_name}"
                    env.BATCH_DATASET_FILENAME = "${variables.batch_dataset_filename}"
                    env.TERRAFORM_STATE_BUCKET = "${variables.terraform_state_bucket}"
                }
                sh '''pip3 install --upgrade --force-reinstall --target $HOME/.local/$MODEL_ID . awscli'''
              }
           }
        stage('Update or Create Train and Inference Sagemaker Pipelines') {
            steps {
                withAWS(roleAccount:"${env.prodAccount}", role:'cross-account-jenkins-role') {
                    sh '''export PATH="$HOME/.local/$MODEL_ID/bin:$PATH"
                          export PYTHONPATH="$HOME/.local/$MODEL_ID:$PYTHONPATH"
                          aws s3 cp pipelines/train/scripts/raw_preprocess.py s3://$DEFAULT_BUCKET/$MODEL_NAME/scripts/raw_preprocess.py 
                          aws s3 cp pipelines/train/scripts/evaluate_model.py s3://$DEFAULT_BUCKET/$MODEL_NAME/scripts/evaluate_model.py 
                          aws s3 cp pipelines/inference/scripts/lambda_helper.py s3://$DEFAULT_BUCKET/$MODEL_NAME/scripts/lambda_helper.py 
                          run-pipeline --module-name pipelines.train.train_pipeline --role-arn $SAGEMAKER_PIPELINE_ROLE --kwargs "{\\"region\\":\\"$REGION\\",\\"role\\":\\"$SAGEMAKER_PIPELINE_ROLE\\",\\"default_bucket\\":\\"$DEFAULT_BUCKET\\",\\"kms_key\\":\\"$KMS_KEY\\",\\"model_artifacts_bucket\\": \\"$MODEL_ARTIFACTS_BUCKET\\",\\"model_name\\": \\"$MODEL_NAME\\",\\"model_package_group_name\\":\\"$MODEL_PACKAGE_GROUP_NAME\\",\\"pipeline_name\\":\\"$TRAIN_PIPELINE_NAME\\"}";
                          run-pipeline --module-name pipelines.inference.inference_pipeline --role-arn $SAGEMAKER_PIPELINE_ROLE --kwargs "{\\"region\\":\\"$REGION\\",\\"role\\":\\"$SAGEMAKER_PIPELINE_ROLE\\",\\"lambda_role\\":\\"$LAMBDA_ROLE\\",\\"default_bucket\\":\\"$DEFAULT_BUCKET\\",\\"kms_key\\":\\"$KMS_KEY\\",\\"model_name\\": \\"$MODEL_NAME\\",\\"model_package_group_name\\":\\"$MODEL_PACKAGE_GROUP_NAME\\",\\"pipeline_name\\":\\"$INFERENCE_PIPELINE_NAME\\",\\"batch_dataset_filename\\":\\"$BATCH_DATASET_FILENAME\\"}";'''
                    
                }
            }
        }
        stage('Terraform init, plan and apply') {
            steps {
                dir("${env.WORKSPACE}/mlops-infra") {
                    withAWS(roleAccount:"${env.prodAccount}", role:'cross-account-jenkins-role') {
                        sh '''terraform init -reconfigure \
                              -backend-config="bucket=$TERRAFORM_STATE_BUCKET" \
                              -backend-config="key=$MODEL_NAME" \
                              -backend-config="region=$REGION" \
                              -input=false'''
                        sh "terraform plan -var-file=../environments/${ENV}.tfvars.json -out tfplan "
                        sh "terraform apply -var-file=../environments/${ENV}.tfvars.json --auto-approve"
                        
                     }
                  }
                }
            }
                
    }
}
