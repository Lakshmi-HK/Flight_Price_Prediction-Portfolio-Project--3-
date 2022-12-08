from sklearn.model_selection import train_test_split

from sklearn import linear_model
from sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import RobustScaler
#from sklearn.preprocessing import SplineTransformer

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

import pickle

from xgboost import XGBRegressor
from catboost import  CatBoostRegressor

def regression(X_train, X_test, y_train, y_test):
    ## Baseline model
    reg = linear_model.LinearRegression()
    reg.fit(X_train,y_train)
    print(metrics.r2_score(y_train,reg.predict(X_train)), metrics.r2_score(y_test,reg.predict(X_test)))
    return reg


def knearestneighbour(X_train, X_test, y_train, y_test):
    knn_regressor = KNeighborsRegressor(n_neighbors=10)
    knn_regressor.fit(X_train, y_train)
    print(metrics.r2_score(y_train,knn_regressor.predict(X_train)),metrics.r2_score(y_test,knn_regressor.predict(X_test)))
    return knn_regressor


def decisiontree(X_train, X_test, y_train, y_test):
    tree = DecisionTreeRegressor(max_depth=10)
    tree.fit(X_train,y_train)
    print(metrics.r2_score(y_train,tree.predict(X_train)),metrics.r2_score(y_test,tree.predict(X_test)))
    return tree


def randomForest(X_train, X_test, y_train, y_test):
    random_regressor = RandomForestRegressor(n_estimators=500, min_samples_split=10,min_samples_leaf=2)
    random_regressor.fit(X_train, y_train)
    print(metrics.r2_score(y_train,random_regressor.predict(X_train)),metrics.r2_score(y_test,random_regressor.predict(X_test)))
    return random_regressor


def xgboost(X_train, X_test, y_train, y_test):
    xgboost_regressor = XGBRegressor()
    xgboost_regressor.fit(X_train, y_train)
    print(metrics.r2_score(y_train,xgboost_regressor.predict(X_train)),metrics.r2_score(y_test,xgboost_regressor.predict(X_test)))
    return xgboost_regressor


def catboost(X_train, X_test, y_train, y_test):
    catboost_regressor = CatBoostRegressor(verbose=False)
    catboost_regressor.fit(X_train, y_train)
    print(metrics.r2_score(y_train,catboost_regressor.predict(X_train)),metrics.r2_score(y_test,catboost_regressor.predict(X_test)))
    return catboost_regressor


def gboost(X_train, X_test, y_train, y_test):
    gboost_regressor = GradientBoostingRegressor()
    gboost_regressor.fit(X_train, y_train)
    print(metrics.r2_score(y_train,gboost_regressor.predict(X_train)),metrics.r2_score(y_test,gboost_regressor.predict(X_test)))
    return gboost_regressor


def train(X, y, modelType):
    # Split your dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    model = modelType(X_train, X_test, y_train, y_test)
    return model


def save_pickle(file_path,obj):
    with open(file_path,'wb') as file:
            pickle.dump(obj,file)

        
def load_pickle(file_path):
    with open(file_path,'rb') as file:
            obj = pickle.load(file)
    return obj