import pandas as pd 
import numpy as np

from time import time
from itertools import product

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection

from sklearn.tree import DecisionTreeClassifier


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix



# load data 
churn = pd.read_excel('churn_prediction_data.xlsx', converters={'Customer_id':str})

# drop row with null value
churn = churn.dropna()

# prepare data for the model
churn['Lost_Active'] = churn['Lost_Active'].replace(['Active', 'Lost'], [1, 0])

churn = churn.drop(columns=['Customer_id'])
churn = pd.get_dummies(churn, prefix=['Customer_Group','Device','Payment','Freq'])

# training and testing split
X = churn.drop('Lost_Active',axis=1)
y = churn['Lost_Active']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30 ,random_state=123, stratify=y)

# Decision Tree
cart = DecisionTreeClassifier(max_depth =4, random_state=0)
cart.fit(X_train, y_train)
y_pred=cart.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))

# Random Forest
clf = RandomForestClassifier(random_state=0)

# use a full grid over all parameters
param_grid = {"n_estimators": [10, 50, 100],
              "max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)
start = time()
grid_search.fit(X_train, y_train)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
report(grid_search.cv_results_)

# look at best classifier 

clf = RandomForestClassifier(n_estimators=50, max_depth=None, max_features= 10, min_samples_leaf= 3, 
                             min_samples_split= 2,bootstrap = True, criterion= 'entropy', random_state=0)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# confusion matrix
print(metrics.classification_report(y_test, y_pred))

# confusion matrix heatmap
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');

# feature importance horizontal bar plot
feature_imp = pd.Series(clf.feature_importances_,index=X.columns).sort_values(ascending=False)
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
