from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.30, random_state=42)

KNN = KNeighborsClassifier(n_neighbors=7)

KNN.fit(x_train, y_train)

print(KNN.predict(x_test))
print(y_test)

dummy = 0
for i in range(len(y_test)):
    if y_test[i] == KNN.predict(x_test)[i]:
        dummy += 1

print(f'Precision: {dummy / (len(y_test))} %')



