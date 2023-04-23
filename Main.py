import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import mlflow.sklearn

from AniketLogisticRegressionModel import AniketLogisticRegressionModel

# Set the experiment name
mlflow.set_experiment("Aniket_Batch_Inference_Test")

# Load the iris dataset
iris = load_iris()

# Create a Pandas dataframe for the feature data and a Pandas series for the target data
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train a logistic regression model
model = AniketLogisticRegressionModel(random_state=42, custom_param1=2.0, custom_param2="bar")
model.fit(X_train, y_train)

# Start an MLflow run
with mlflow.start_run():

    # Log the model parameters
    mlflow.log_param("C", model.C)
    mlflow.log_param("penalty", model.penalty)
    mlflow.log_param("random_state", model.random_state)
    mlflow.log_param("custom_param1", model.custom_param1)

    # Log the model metrics
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    mlflow.log_metric("accuracy", accuracy)

    # Log the model itself
    mlflow.sklearn.log_model(model, "model")

    # Print the MLflow tracking URI
    print(mlflow.get_tracking_uri())

