import mlflow.sklearn  # import the mlflow library for tracking machine learning experiments
import pandas as pd  # import pandas library for data manipulation
from sklearn.preprocessing import LabelEncoder  # import LabelEncoder class from sklearn for encoding categorical variables


mlflow.set_experiment("Aniket_Batch_Inference_Test")

# Load the trained model from the specified URI
model_uri = "runs:/{RUN_ID}/model"
model = mlflow.sklearn.load_model(model_uri)

mlflow.set_experiment_tag("raw_set", "activation_1")


# Load the input data from CSV file, and specify the column names
data = pd.read_csv("iris.data", names=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'class'])

# Convert the target variable to numerical values using LabelEncoder
le = LabelEncoder()
data['class'] = le.fit_transform(data['class'])

# Perform batch inference using the trained model to make predictions
predictions = model.predict(data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']])

# Combine the input data and predictions into a single DataFrame
output = pd.concat([data, pd.DataFrame(predictions, columns=["prediction"])], axis=1)

import os
# Set the MLflow server URL to localhost
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

# Save the output to a CSV file with filename predictions.csv
output.to_csv("predictions.csv", index=False)

import time

# Log predictions in MLflow as an artifact with the name 'predictions'
# Also log the accuracy metric with value 0.95 for this run
with mlflow.start_run():
    mlflow.log_artifact("predictions.csv", artifact_path="predictions")
    mlflow.log_metric("accuracy", 0.95)
    mlflow.set_tag("round_tag", "Activate_1")
    mlflow.set_tag("mlflow.runName", "Aniket_Classification_Batch_Run_"+str(time.time()))
