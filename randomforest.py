import pickle as achar
import sklearn.ensemble as ske
import numpy as np
import sklearn.svm as sksvm
import sklearn.metrics as skm
import xgboost as xgb
import sklearn.neighbors as skn
import sklearn.tree as skt
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.discriminant_analysis as skd


def load_achar(filename):
    fileobj = open(filename, 'rb')
    print(fileobj)
    obj = achar.load(fileobj, encoding='latin1')
    fileobj.close()
    return obj

X_train = load_achar('X_train')
X_test = load_achar('X_test')
y_train = load_achar('y_train')
y_test = load_achar('y_test')

for i in range(len(X_train)):
    for j in range(len(X_train[i])):
        X_train[i][j] = 10.0 if X_train[i][j] > 10 else float(X_train[i][j])

for i in range(len(X_test)):
    for j in range(len(X_test[i])):
        X_test[i][j] = 10.0 if X_test[i][j] > 10 else float(X_test[i][j])

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

# ske.RandomForestClassifier(n_estimators=800, random_state=0) 87%
# xgb.XGBClassifier(objective="binary:logistic", random_state=0) 87%
# skn.KNeighborsClassifier(random_state=0) 85%
# skt.DecisionTreeClassifier(random_state=0) 83%
# MLPClassifier() 86%
# ske.AdaBoostClassifier(n_estimators=800) 85%
# ske.GradientBoostingClassifier(n_estimators=800) 86%
# LogisticRegression() 85%
# skd.LinearDiscriminantAnalysis() 85%

model = ske.RandomForestClassifier(n_estimators=800, random_state=0)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(skm.confusion_matrix(model.predict(X_test), y_test))