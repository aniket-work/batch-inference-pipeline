## Batch Inference Pipeline

### Overview

This pipeline though described for a simple example, can be designed to process a large volume of data, technique is for processing of multiple inputs at once. The pipeline takes in input data, performs batch inference using a trained machine learning model, and outputs the results to a specified location.

This simple project trains a logistic regression model using the iris dataset, splits the data into training and testing sets, logs the model parameters and metrics to MLflow, and saves the trained model to the MLflow local registry.


![alt text](https://github.com/aniket-work/batch-inference-pipeline/blob/master/flow_arch.png?raw=true)

### Prerequisites

Before we begin, make sure you have the following installed:

 - Python 3.6 or higher
 - MLflow (pip install mlflow)
 - Scikit-learn (pip install scikit-learn)
 - Pandas (pip install pandas)

### Usage
 - Download dataset
 ```bash:
  curl https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data -o iris.data
```
 
 - Run the script using.
  ```python
   python Main.py
  ```
   
    Note - 
    * The script will output the MLflow tracking URI to the console.
    * use {RUN_ID} from the mlFlow deploy model 

### Generic flow overview

1. **Input Data Preparation:** Prepare the input data to be processed. This may include data cleaning, formatting, and preprocessing as required by the model.

2. **Batch Inference:** Use the trained machine learning model to perform batch inference on the input data. This can be done using a batch inference library or by implementing a custom batch inference function.

3. **Output Results:** Save the results of the batch inference to a specified location, such as a file or database.

### Usage

To use this pipeline, follow these steps:

1. Clone the repository.
2. Prepare your input data as required by the model.
3. Modify the pipeline script to specify the location of the input data and desired output location.
4. Run the pipeline script.

### Code overview
 - Load the iris dataset using sklearn.datasets.load_iris.
 - Split the dataset into training and test sets using sklearn.model_selection.train_test_split.
 - Train a logistic regression model using the custom AniketLogisticRegressionModel class.
 - Start an MLflow run using mlflow.start_run().
 - Log the model parameters to MLflow using mlflow.log_param.
 - Log the model metrics to MLflow using mlflow.log_metric.
 - Save the trained model to the MLflow registry using mlflow.sklearn.log_model.
 - Print the MLflow tracking URI to the console using mlflow.get_tracking_uri()
