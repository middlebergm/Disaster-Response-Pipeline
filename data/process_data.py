import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Ingests 2 data sets and joins the 2 together into 1 dataset
    input:
        messages
        categories
    output:
        merged dataset
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='left',on='id')
    return df

def clean_data(df):
    '''
    ingests merged dataframe and splits categories into separate columns. If the value in the column is 2, replace it with 1. 
    Returns clean dataset.
    input:
        dataframe
    output:
        df that has been cleaned
    '''
    # split categories into new columns
    categories = df['categories'].str.split(';',expand=True)
    #get names of categories and assign them as column names of dataframe
    row = list(categories.iloc[0].values)
    category_colnames = list(map(lambda x: x.split('-')[0], row))
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-',expand=True)[1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
        # change any values of 2 to 1
        categories.loc[categories[column]==2,column]=1
    
    # drop original "categories" column from dataset
    df.drop('categories',axis=1,inplace=True)
    
    # combine df and cleaned categories columnes
    df = pd.concat([df,categories],axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    '''
    takes in the merged/cleaned dataframe and loads the data into a table within a SQL database
    input:
        df
        database to save the data to
    output:
        None
    '''
    # create engine for SQLite
    engine = create_engine('sqlite:///'+ database_filename)
    # save df to database with new filename
    df.to_sql('cleandata', engine, index=False)
    
    return None


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()