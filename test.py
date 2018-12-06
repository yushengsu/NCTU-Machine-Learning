import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import os
if not os.path.exists('JPG'):
    os.makedirs('JPG')

def concrete_process():
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
        
        plt.scatter(X,Y,color='red')
        plt.title(input_col[index])
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig('JPG/' + input_col[index] + '.png')
        plt.close()

    x_train = x_train.iloc[:, [0]].values
    y_train = y_train.iloc[:, [0]].values
    x_test = x_test.iloc[:, [0]].values
    y_test = y_test.iloc[:, [0]].values
    
    linear_model = LinearRegression()
    linear_model.fit(x_train,y_train)

    y_pred = linear_model.predict(x_test)
    
    print'Name:', input_col[0])
    print('Weight:', linear_model.coef_[0][0])
    print('Bias:', linear_model.intercept_[0])
    print('Accuracy:', r2_score(y_test,y_pred))
    

    plt.scatter(x_test,y_test,color='red')
    plt.plot(x_test,y_pred,color='blue')
    plt.title(" (Test data set)")
    plt.xlabel("x_test")
    plt.ylabel("y_test")
    plt.savefig('JPG/' + input_col[0] + '_regression.png')

def main():
    concrete_process()
if __name__ == '__main__':
    main()