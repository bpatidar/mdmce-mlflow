import heapq
import mlflow
import joblib
import pandas as pd

# Define the model class
class TextCategorizerModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        store = joblib.load(context.artifacts["classifier_model"])
        ## Load the classifier and labelencoder from the stored file
        self.le = store[0]
        self.clf = store[1]

    ## Mandatory predict() implementation.
    def predict(self, context, test_data):

        #ids, descriptions = preprocess(test_data)
        ids = test_data['pkey']
        descriptions = test_data['description']

        res = self.clf.predict_proba(descriptions)
        ## Is a list of list
        reslist = []
        n=3
        for entry in res:
            print("Result")
            print(entry)
            ## Fetch the top n recommended categories and confindences
            ncategories = heapq.nlargest(n, enumerate(entry), key=lambda x: x[1])
            cat_code, confidence = zip(*ncategories)

            ## Decode the predicted klass to the Text Category
            category = self.le.inverse_transform(list(cat_code))

            ## Zip the result tuple and add it to the result list
            result = [category, list(confidence)]
            print(result)
            reslist.append(result)

        data = {'pkey':ids, 'description':descriptions, 'categories': reslist}

        df = pd.DataFrame(data)
        data = df.to_json(orient='split', index=False)

        return data