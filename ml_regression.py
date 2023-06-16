# random forest regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

data = pd.read_parquet('data_cleaned.parquet')
data = data.drop(['cat_Ventes'], axis=1)
X = data.drop(['Ventes'], axis=1)
y = data['Ventes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42069)

reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print(mean_squared_error(y_test, y_pred))

scores = cross_val_score(reg, X, y, cv=5)
print(scores)
print(np.mean(scores))

