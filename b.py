

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
irisdata=load_iris()

a = irisdata.data
b = irisdata.target
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.3, random_state=42)

knn = KNeighborsClassifier()
knn.fit(a_train, b_train)
c = knn.predict(a_test)
acc = accuracy_score(b_test, c)
print(c)
print('ACCURACY:',acc)
