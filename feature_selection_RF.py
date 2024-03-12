import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
sns.set()
plt.figure(figsize=(18,10))
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
df = pd.read_csv(url, header = None)
df.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
              'Alcalinity of ash', 'Magnesium', 'Total phenols',
              'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
              'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
X = df.drop('Class label', axis=1)
y = df['Class label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
feat_labels = df.columns[1:]
clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
clf.fit(X_train, y_train)
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
indices = indices[::-1]
plt.barh(feat_labels[indices], width=clf.feature_importances_[indices], height=0.5)
plt.xlabel("Random Forest Feature Importance")
plt.savefig("features selection.jpg", dpi=600)
plt.show()
