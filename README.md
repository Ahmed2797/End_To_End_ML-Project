<<<<<<< HEAD
# This is my first ML_projects End_to_End
=======

>>>>>>> 87fef144cc31537f551a80ef0696d2df0f9cbd1c

import dagshub
dagshub.init(repo_owner='Ahmed2797', repo_name='End_To_End_ML-Project', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)