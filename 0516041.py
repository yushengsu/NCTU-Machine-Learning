from sklearn import datasets
from sklearn.model_selection import train_test_split

import pydotplus
import collections

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import export_graphviz

iris = datasets.load_iris()
x = pd.DataFrame(iris['data'], columns=iris['feature_names'])
#print("target_names: "+str(iris['target_names']))
y = pd.DataFrame(iris['target'], columns=['target'])
iris_data = pd.concat([x,y], axis=1)
data_feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal width (cm)', 'petal length (cm)']
x_train, x_test, y_train, y_test = train_test_split(iris_data[data_feature_names], iris_data[['target']], test_size=0.3, random_state=0)

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = 'entropy', max_depth=3, random_state=0)
tree = tree.fit(x_train,y_train)

dot_data = export_graphviz(tree, out_file = None, feature_names=data_feature_names)

graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('tree.png')

# print(tree.predict(x_test))

# print(y_test['target'].values)

# print(tree.score(x_test,y_test['target']))

