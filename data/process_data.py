#Import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

#Load and merge datasets
def load_data(messages_filepath, categories_filepath):
    """
    Load Messages and categories Data to merge
    
    Arguments:
        messages_filepath -> Path to the CSV file containing messages
        categories_filepath -> Path to the CSV file containing categories
    Output:
        df -> Combined data containing messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, left_on='id', right_on='id', how='left')
    return df

#Clean the merged dataset
def clean_data(df):
    """
    Clean df dataset
    
    Arguments:
        df -> Combined data containing messages and categories
    Outputs:
        df -> Combined data containing messages and categories with categories cleaned up
    """
    categories = df.categories.str.split(pat=';', expand=True)
    row = categories.head(1)
    category_colnames = (row.apply(lambda x: x.str[:-2]).values.tolist())
    category_colnames = category_colnames[0]
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(int)
        
    df.drop(columns=['categories'], axis=1, inplace=True)
    
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    
    return df

#Save the cleaned dataset into a db file
def save_data(df, database_filename):
    """
    Save dataset to SQLite Database
    
    Arguments:
        df -> Combined data containing messages and categories with categories cleaned up
        database_filename -> Path to SQLite database
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace')  

#Execute the pipeline
def main():
    """
    Main function which will start the data processing functions. It follows the actions taken after starting this function:
        1) Load Messages and Categories data
        2) Clean the Data
        3) Save Data to SQLite Database
    """
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