import pandas as pd
##### What module decision tree need #####
from sklearn import datasets
from sklearn.model_selection import train_test_split

import pydotplus
import collections

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import export_graphviz
##### What module decision tree need #####

def googleplay_preprocess():
    df = pd.read_csv('googleplaystore.csv')
    df2 = pd.read_csv('googleplaystore_user_reviews.csv')

    df['Rating'].fillna(value=df.groupby('Category')['Rating'].transform('mean'), inplace=True)
    df['Installs'] = df['Installs'].apply(lambda x: x.replace(',', ''))
    df['Installs'] = df['Installs'].apply(lambda x: x.replace('+', ''))
    df = df[df['Installs'] != 'Free']
    df['Installs'] = df['Installs'].astype('int')
    df = df[['App', 'Category', 'Rating', 'Reviews', 'Installs']]
    df2 = df2[['App', 'Sentiment_Polarity', 'Sentiment_Subjectivity']]
    df2['Sentiment_Polarity'] = df2.groupby('App')['Sentiment_Polarity'].transform('mean')
    df2['Sentiment_Subjectivity'] = df2.groupby('App')['Sentiment_Subjectivity'].transform('mean')

    df = df.drop_duplicates('App')
    df = df.reset_index(drop=True)
    df2 = df2.drop_duplicates('App')
    df2 = df2.reset_index(drop=True)
    result = pd.concat([df.set_index('App'), df2.set_index('App')], axis=1, sort=False)
    result = result.reset_index()
    result = result.rename(columns={'index': 'App'})
    result = result.dropna()
    result = result.reset_index(drop=True)

    return result

def decision_tree():
    result = googleplay_preprocess()
    #print(result)
    
    data_feature_names = ['Reviews', 'Sentiment_Polarity', 'Sentiment_Subjectivity']
    x_train, x_test, y_train, y_test = train_test_split(result[data_feature_names], result[['Rating']], test_size=0.3, random_state=0)

    from sklearn.tree import DecisionTreeRegressor
    tree = DecisionTreeRegressor()
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


def main():
    decision_tree()


if __name__ == '__main__':
    main()
