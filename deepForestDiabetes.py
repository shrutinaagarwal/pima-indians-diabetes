from gcForest import gcForest
from sklearn.datasets import load_iris, load_digits
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import urllib.request
 
# URL for the Pima Indians Diabetes dataset (UCI Machine Learning Repository)
url = "http://goo.gl/j0Rvxq"
# download the file
raw_data = urllib.request.urlopen(url)
# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, delimiter=",")
print(dataset.shape)
# separate the data from the target attributes
X = dataset[:,0:7]
y = dataset[:,8]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4)

#gcf = gcForest(shape_1X=4, window=2, tolerance=0.0, min_samples_mgs=20, min_samples_cascade=10)
#gcf.fit(X_tr, y_tr)

#pred_X = gcf.predict(X_te)
#print(pred_X)

gcf = gcForest(tolerance=0.0, min_samples_mgs=40, min_samples_cascade=20)
_ = gcf.cascade_forest(X_tr, y_tr)

pred_proba = gcf.cascade_forest(X_te)
tmp = np.mean(pred_proba, axis=0)
preds = np.argmax(tmp, axis=1)


# evaluating accuracy
#accuracy = accuracy_score(y_true=y_te, y_pred=pred_X)
print('gcForest accuracy using multigrain scaning: {}'.format(accuracy_score(y_true=y_te, y_pred=preds)))



gcf = gcForest(tolerance=0.0, min_samples_cascade=20)
_ = gcf.cascade_forest(X_tr, y_tr)
pred_proba = gcf.cascade_forest(X_te)
tmp = np.mean(pred_proba, axis=0)
preds = np.argmax(tmp, axis=1)
accuracy_score(y_true=y_te, y_pred=preds)
print('gcForest accuracy without multigrain scanning : {}'.format(accuracy_score(y_true=y_te, y_pred=preds)))







