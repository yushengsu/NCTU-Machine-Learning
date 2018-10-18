import pandas as pd
##### What module decision tree need #####
from sklearn import datasets
from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype

import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
sns.set(style="white", color_codes=True)
import numpy as np
import matplotlib.pyplot as plt
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

def remove_outlier(df):
    low = .05
    high = .95
    quant_df = df.quantile([low, high])
    for name in list(df.columns):
        if is_numeric_dtype(df[name]):
            df = df[(df[name] > quant_df.loc[low, name]) & (df[name] < quant_df.loc[high, name])]
    return df

def visualization():
    result = googleplay_preprocess()
    #print(result)
    data_feature_names = ['Sentiment_Polarity', 'Sentiment_Subjectivity']
    
    x = result[data_feature_names].astype(np.float)
    y = result[['Rating']].round(0).astype(int)

    result_data = pd.concat([x,y], axis=1)   
    result_data.boxplot(by="Rating", figsize=(10, 10))


    plt.savefig('Rating.png', bbox_inches='tight')
 
    plt.close()

    x.loc['avg'] = x.mean()
    x.loc['std'] = x.std()
    print("average:")
    print(x.loc['avg'] )
    print(" ")
    print("standard deviation:")
    print(x.loc['std'] )
    print(" ")

    x['Sentiment_Polarity'].hist(bins = 30)
    plt.title("Sentiment_Polarity")
    plt.savefig('Sentiment_Polarity.png')
    plt.close()


    x['Sentiment_Subjectivity'].hist(bins = 30)
    plt.title("Sentiment_Subjectivity")
    plt.savefig('Sentiment_Subjectivity')
    plt.close()



def main():
    visualization()


if __name__ == '__main__':
    main()
