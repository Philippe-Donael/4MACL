from lazypredict.Supervised import LazyRegressor
from sklearn import datasets
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_parquet('data_cleaned.parquet')
data = data.drop(['cat_Ventes'], axis=1)
X = data.drop(['Ventes'], axis=1)
y = data['Ventes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42069)

reg = LazyRegressor(verbose=1, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models)
print(predictions)
model_dictionary = reg.provide_models(X_train, X_test, y_train, y_test)
print(model_dictionary)
