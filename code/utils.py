import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
import numpy as np
import os
"""
Load the training data.
"""
def fetch_data():
    df = pd.read_excel("data/MDM_Mckesson_data.xlsx", usecol=[0, 1], header=0, names=['category', 'description'])
    df['description'] = df['description'].apply(lambda x: x.lower())
    df.drop_duplicates(inplace=True)
    return df

"""
Transforms the text catgeories into to a number (list of unique category)
"""
def encode_categories(categories_list):
    le = preprocessing.LabelEncoder()
    Y = le.fit_transform(categories_list)
    klasses = le.transform(categories_list)
    return klasses, le

"""
Use GridsearchCV to create the pipeline and choose the best estimator model.
"""
def get_model(df, C=1.0, kernel="rbf"):
    combined_features = FeatureUnion([
        ("tfidf_word", TfidfVectorizer(analyzer='word')),
        ("tfidf_char", TfidfVectorizer(analyzer='char'))
    ])

    pipeline = Pipeline([("features", combined_features),
                         ('clf', OneVsRestClassifier(svm.SVC(probability=True)))])

    params_ovr = dict(clf__estimator__C=[C], clf__estimator__class_weight=['balanced'],
                      clf__estimator__kernel=[kernel],
                      features__tfidf_word__ngram_range=[(1, 3)], features__tfidf_char__ngram_range=[(2, 3)],
                      features__tfidf_word__min_df=[2], features__tfidf_char__min_df=[2],
                      features__tfidf_word__max_features=[500], features__tfidf_char__max_features=[200]
                      )
    grid_search = GridSearchCV(pipeline, param_grid=params_ovr, verbose=10, n_jobs=-1)

    grid_search.fit(df.description.values, df.klass.values)
    print("Best score: %0.3f" % grid_search.best_score_)

    return grid_search.best_estimator_, grid_search.best_score_

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2