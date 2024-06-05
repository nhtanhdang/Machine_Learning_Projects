## READ ME ##
# This model employs batch learning method in production 
import os 
import tarfile
import urllib
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_squared_error

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, RandomizedSearchCV, cross_val_score 
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.ensemble import RandomForestRegressor

from zlib import crc32

# Get the path to the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join(PROJECT_ROOT, "datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# fetch data 
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    if not os.path.exists(tgz_path):
        urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

# load data from csv file
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

# split data to train & test set using stratified sampling
def split_train_test(data, col):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data[col]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]
    return strat_train_set, strat_test_set

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing to do in this example
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix] # bedroom-room ratio
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
            # np.c_ concat series/2d arrays along columns (axis=1)
        else:
            return np.c_[X, rooms_per_household, population_per_household]

def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class AttributesSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self 
    def transform(self, X):
        return X[:, self.feature_indices_]

def display_score(scores):
    print("Cross Validation\n")
    print("Errors: ", scores)
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())

    
def evaluate_with_rmse(model, X, y):
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    return rmse

def main():
    # fetch and load data
    fetch_housing_data()
    housing = load_housing_data()

    # split data to train & test set using stratified sampling
    housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3., 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
    strat_train_set, strat_test_set = split_train_test(housing, "income_cat")

    # remove the income_cat to return dataset to its original state
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True) 
    
    # separate predictors and label
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    # separate numeric and categorical features
    housing_num = housing.drop("ocean_proximity", axis=1)
    housing_cat = housing[["ocean_proximity"]]

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    # pipeline to preprocess numerical features
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

    # full pipeline for all features
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs), # return dense matrix
        ("cat", OneHotEncoder(), cat_attribs), # return sparse matrix
    ], remainder="drop")

    # prepare data
    housing_prepared = full_pipeline.fit_transform(housing)

    # initialise model
    forest_reg = RandomForestRegressor()

    # first train without fine-tune
    forest_reg.fit(housing_prepared, housing_labels)
    train_rmse = evaluate_with_rmse(forest_reg, housing_prepared, housing_labels)
    print("Training Error: ", train_rmse)

    # cross validation
    scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                        scoring="neg_mean_squared_error", cv=10)
    forest_rmse_scores = np.sqrt(-scores)
    display_score(forest_rmse_scores)


    # initialise hyperparam for random search
    param_search = [
        {'n_estimators': [1, 10, 100], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators':[1, 10, 100], 'max_features': [2, 3, 4]},
    ]

    # random search
    ran_search = RandomizedSearchCV(forest_reg, param_search, cv=5,
                                    scoring='neg_mean_squared_error',
                                    return_train_score=True)

    ran_search.fit(housing_prepared, housing_labels)

    feature_importances = ran_search.best_estimator_.feature_importances_
    best_params = ran_search.best_params_
    best_model = ran_search.best_estimator_
    best_score = ran_search.best_score_
    print("Random Search Error: ", np.sqrt(-best_score))

    # select top 5 important attributes 
    selector = AttributesSelector(feature_importances, 5)
    housing_selected = selector.fit_transform(housing_prepared)

    # retrain model on top 5 important attributes and best params from random search
    forest_reg_v2 = RandomForestRegressor(**best_params)
    forest_reg_v2.fit(housing_selected, housing_labels)
    train_v2_rmse = evaluate_with_rmse(forest_reg_v2, housing_selected, housing_labels)
    print("Training v2 Error: ", train_v2_rmse)

    # cross validation
    scores_v2 = cross_val_score(forest_reg_v2, housing_selected, housing_labels,
                        scoring="neg_mean_squared_error", cv=10)
    forest_v2_rmse_scores = np.sqrt(-scores_v2)
    display_score(forest_v2_rmse_scores)

    # TEST, without feature selection
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    X_test_prepared = full_pipeline.transform(X_test)
    test_rmse = evaluate_with_rmse(best_model, X_test_prepared, y_test)
    print("Test Error: ", test_rmse)


if __name__ == "__main__":
    main()



