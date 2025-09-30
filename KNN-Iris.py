from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt


iris = datasets.load_wine()

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.30, random_state=42)

KNN = KNeighborsClassifier(n_neighbors=7)

KNN.fit(x_train, y_train)

print(KNN.predict(x_test))
print(f'Precision KNN: {KNN.score(x_test, y_test)} %')

DecisionTree = DecisionTreeClassifier()

DecisionTree.fit(x_train, y_train)

print(DecisionTree.predict(x_test))
print(f'Precision Decision Tree: {DecisionTree.score(x_test, y_test)} %')

print(tree.plot_tree(DecisionTree.fit(iris.data, iris.target)))
