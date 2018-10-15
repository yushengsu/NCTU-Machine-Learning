from sklearn import datasets
from sklearn.model_selection import train_test_split

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


sns.set(style="white", color_codes=True)

#To import the Iris dataset:
iris = datasets.load_iris()
x = pd.DataFrame(iris['data'], columns=iris['feature_names'])

y = pd.DataFrame(iris['target'], columns=['species'])

for index in y.index:
    if y['species'][index] == 0:
    	y['species'][index] = "Iris-setosa"

    elif y['species'][index] == 1:
    	y['species'][index] = "Iris-versicolor"
    else:
    	y['species'][index] = "Iris-virginica"
iris_data = pd.concat([x,y], axis=1)

iris_data.boxplot(by="species", figsize=(10, 10))
plt.savefig('species.png', bbox_inches='tight')
plt.close()

x.loc['avg'] = x.mean()
x.loc['std'] = x.std()
print("average:")
print(x.loc['avg'] )
print(" ")
print("standard deviation:")
print(x.loc['std'] )
print(" ")

x['sepal length (cm)'].hist(bins = 30)
plt.title("sepal length (cm)")
plt.savefig('sepal_length.png')
plt.close()


x['sepal width (cm)'].hist(bins = 30)
plt.title("sepal width (cm)")
plt.savefig('sepal_width.png')
plt.close()


x['petal length (cm)'].hist(bins = 30)
plt.title("petal length (cm)")
plt.savefig('petal_length.png')
plt.close()


x['petal width (cm)'].hist(bins = 30)
plt.title("petal width (cm)")
plt.savefig('petal_width.png')
plt.close()
#To view Iris data below: