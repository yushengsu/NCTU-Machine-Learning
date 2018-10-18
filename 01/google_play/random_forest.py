from sklearn import datasets
import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


def random_forest(x_train, y_train, x_test, y_test, n_estimators=100,
                  resubstitution=False, k_fold_cv=False, cv=5,
                  confusion=False):
    '''
    :param x_train: type: pd.DataFrame()
    :param y_train: type: pd.DataFrame(), the column should be named 'target'
    :param x_test: type: pd.DataFrame()
    :param y_test: type: pd.DataFrame(), the column should be named 'target'
    :param n_estimators:
    :return: the score
    '''
    if k_fold_cv:
        x = pd.concat([x_train, x_test]).reset_index(drop=True)
        y = pd.concat([y_train, y_test]).reset_index(drop=True)
        return k_fold_cross_validation(x, y, cv)
    if resubstitution:
        x = pd.concat([x_train, x_test]).reset_index(drop=True)
        y = pd.concat([y_train, y_test]).reset_index(drop=True)
        return random_forest(x, y, x, y)
    vote = pd.DataFrame(columns=range(0, len(x_test)))
    elected = []
    for x in range(n_estimators):
        random_list = [random.randint(0, len(x_train)-1) for _ in range(len(x_train))]
        x_tree = x_train.loc[random_list]
        y_tree = y_train.loc[random_list]
        print(y_tree)
        tree = DecisionTreeClassifier(criterion='entropy', splitter='random')
        tree.fit(x_tree, y_tree.astype(int))
        if not resubstitution:
            vote.loc[x] = (tree.predict(x_test).tolist())
        else:
            vote.loc[x] = (tree.predict(x_train).tolist() + tree.predict(x_test).tolist())
    for x in range(len(x_test)):
        elected.append(vote[x].value_counts().index.tolist()[0])
    if confusion:
        C = confusion_matrix(y_test, elected)
        return C / C.astype(np.float).sum(axis=1)
    correct = 0
    for x in range(len(x_test)):
        if elected[x] == y_test.loc[x]['target']:
            correct += 1
    return correct/len(x_test)


def k_fold_cross_validation(x, y, cv=5):
    l = [i for i in range(len(x))]
    n = int(len(x)/cv)
    split = tuple(l[i:i + n] for i in range(0, len(l), n))
    score = []
    for k_fold in split:
        x_test = pd.DataFrame()
        x_train = pd.DataFrame()
        y_test = pd.DataFrame()
        y_train = pd.DataFrame()
        for tmp in split:
            if tmp != k_fold:
                x_train = pd.concat([x_train, x.loc[tmp]])
                y_train = pd.concat([y_train, y.loc[tmp]])
            else:
                x_test = pd.concat([x_test, x.loc[tmp]])
                y_test = pd.concat([y_test, y.loc[tmp]])
        x_train = x_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        x_test = x_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        score.append(random_forest(x_train, y_train, x_test, y_test))
    return sum(score)/len(score)


def iris_random_forest():
    iris = datasets.load_iris()
    x = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    y = pd.DataFrame(iris['target'], columns=['target'])
    iris_data = pd.concat([x, y], axis=1)
    iris_data['is_train'] = np.random.uniform(0, 1, len(iris_data)) <= .7
    train, test = iris_data[iris_data['is_train']], iris_data[iris_data['is_train'] == False]
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    x_train, y_train = train[iris['feature_names']], pd.DataFrame(train['target'])
    x_test, y_test = test[iris['feature_names']], pd.DataFrame(test['target'])
    print('random forest:', random_forest(x_train, y_train, x_test, y_test, 100))
    print('random forest(resubstitution validation):', random_forest(x_train, y_train, x_test, y_test, 100, resubstitution=True))
    print('random forest(K-fold cross validation):', random_forest(x_train, y_train, x_test, y_test, 100, k_fold_cv=True))
    c = (random_forest(x_train, y_train, x_test, y_test, 100, confusion=True))
    species = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    df_c = pd.DataFrame(c, species, species)
    print('random forest(confusion matrix):\n', df_c)
    sns.set(font_scale=1)
    sns.heatmap(df_c, annot=True, annot_kws={'size': 14}, cmap='YlGnBu')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig('iris_confusion_matrix.png')


def main():
    pass


if __name__ == '__main__':
    main()
