import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import xgboost

import os
if not os.path.exists('Jpg'):
    os.makedirs('Jpg')
if not os.path.exists('Regression'):
    os.makedirs('Regression')

def Regression_with_single():
    data = pd.read_csv("Concrete_Data.csv")
    input_col = list(data)
    output_col = input_col[8]

    x = data[input_col]
    y = data[output_col]
    concrete_data = pd.concat([x, y], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(concrete_data[input_col], concrete_data[output_col], test_size=0.2, random_state=0)
    
    for index in range(0, 8):
        X = x_train.iloc[:, [index]].values
        Y = y_train.iloc[:, [0]].values
        input_col[index] = input_col[index].split() 
        plt.scatter(X,Y,color='red')
        plt.title(input_col[index][0])
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig('Jpg/' + input_col[index][0] + '.png')
        plt.close()
    
        X_train = x_train.iloc[:, [index]].values
        Y_train = y_train.iloc[:, [0]].values
        X_test = x_test.iloc[:, [index]].values
        Y_test = y_test.iloc[:, [0]].values
        
        linear_model = LinearRegression()
        linear_model.fit(X_train,Y_train)

        Y_pred = linear_model.predict(X_test)
        
        print('Name:', input_col[index][0])
        print('Weight:', linear_model.coef_[0][0])
        print('Bias:', linear_model.intercept_[0])
        print('r2_score:', r2_score(Y_test, Y_pred), '\n')

        plt.scatter(X_test,Y_test,color='red')
        plt.plot(X_test,Y_pred,color='blue')
        plt.title(' (' + input_col[index][0] + '_regression set)')
        plt.xlabel('x_test')
        plt.ylabel('y_test')
        plt.savefig('Regression/' + input_col[index][0] + '_regression.png')

def Diffrent_regression():
    data = pd.read_csv("Concrete_Data.csv")
    input_col = list(data)[0:8]
    output_col = list(data)[8]
    
    x = data[input_col]
    y = data[output_col]
    concrete_data = pd.concat([x, y], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(concrete_data[input_col], concrete_data[output_col],test_size=0.2)
    xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                               colsample_bytree=1, max_depth=7)
    xgb.fit(x_train,y_train)
    predictions = xgb.predict(x_test)
    print('r2_score: ', r2_score(y_test, predictions))

def main():
    Regression_with_single()
    Diffrent_regression()
if __name__ == '__main__':
    main()
