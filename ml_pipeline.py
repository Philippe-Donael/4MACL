from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


data = pd.read_parquet('data_cleaned.parquet')

data = data.drop(['Ventes'], axis=1)
X = data.drop(['cat_Ventes'], axis=1)
y = data['cat_Ventes']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42069)

clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

scores = cross_val_score(clf, X, y, cv=5)
print(scores)
print(np.mean(scores))

# confusion matrix using seaborn with labels
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=['0', '1', '2'], yticklabels=['0', '1', '2'])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

# roc curve
y_pred_proba = clf.predict_proba(X_test)[::, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba, pos_label=2)
plt.plot(fpr, tpr)
plt.show()

