"""
This trains a ML model reading all possible variations of product descriptions and their respective categories.

Expects an xlsx file containing category as first column and product description as second.
"""

import mlflow
import mlflow.pyfunc
import warnings
import os
from sklearn.model_selection import train_test_split
import numpy as np
from utils import *
import joblib
from text_categorizer_model import TextCategorizerModel

def train(C, kernel='linear', test_size=0.2):

    np.random.seed(40)
    global df, le
    train_data, test_data = train_test_split(df, test_size=test_size)

    with mlflow.start_run():
        ## Fetch the best model for the data.
        clf, score = get_model(train_data, C, kernel)

        actual = le.transform(test_data['category'])
        predicted = clf.predict(test_data['description'])
        (rmse, mae, r2) = eval_metrics(actual, predicted)

        mlflow.log_param('C', C)
        mlflow.log_param('kernel', kernel)
        mlflow.log_param('test_size', test_size)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("classifier score", score)

        ## Create list of items need to store in the file
        store = [le, clf]

        runid = mlflow.active_run().info.run_id
        path_to_model = "model/classifier-"+runid+".joblib"

        ## Store the model into file
        joblib.dump(store, path_to_model)

        artifacts = {
            "classifier_model": path_to_model
        }

        mlflow_pyfunc_model_path = "generated/mck_mlflow_pyfunc_"+kernel+"_"+str(C)+"_"+runid
        ## Saves in a local dir specified by path param
        mlflow.pyfunc.save_model(
            path=mlflow_pyfunc_model_path, python_model=TextCategorizerModel(), artifacts=artifacts, conda_env="conda.yaml" )

        ## logs as artifact on the mlflow runs.
        mlflow.pyfunc.log_model(artifact_path=mlflow_pyfunc_model_path, python_model=TextCategorizerModel(), artifacts=artifacts, conda_env="conda.yaml")

        print("Saving the model")



import sys
if __name__ =="__main__":
    warnings.filterwarnings("ignore")
    ## test.
    test = False
    if test:
        # Load the model in `python_function` format
        loaded_model = mlflow.pyfunc.load_model("generated/mck_mlflow_pyfunc_linear_1")

        data = {"pkey": ["ID003"],
                "description": ["Ear Curette McKesson Single-ended Handle with Grooves 4 mm Tip Oval Tip"]
                }

        import pandas as pd

        # Create DataFrame
        df = pd.DataFrame(data)
        data = df.to_json(orient='split', index=False)
        test_predictions = loaded_model.predict(df)
        print("Result is :: ")
        print(test_predictions)

    else:

        ## Load the train data and prepare the target categories into a model consumable way
        df = fetch_data()
        df['klass'], le = encode_categories(df.category)

        ## Fetch the arguments/ hyperparameter values passed to the program
        C = float(sys.argv[1])
        kernel = sys.argv[2]
        test_size = float(sys.argv[3])

        train(C, kernel, test_size)
