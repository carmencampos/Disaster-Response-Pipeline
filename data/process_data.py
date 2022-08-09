# python process_data.py messages.csv categories.csv DisasterMessages.db

# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages, categories):
  """
  load messages dataset
  load categories dataset
  merge datasets
  """
  messages = pd.read_csv(messages)
  categories = pd.read_csv(categories)

  df = pd.merge(messages, categories)
  
  return df


def clean_data(df):
  """
  create a dataframe of the 36 individual category columns
  select the first row of the categories dataframe
  use this row to extract a list of new column names for categories.
  one way is to apply a lambda function that takes everything 
  up to the second to last character of each string with slicing
  rename the columns of `categories`
  drop the original categories column from `df`
  concatenate the original dataframe with the new `categories` dataframe
  drop duplicates
  """
  df = pd.DataFrame(df)
  categories = df['categories'].str.split(';', expand=True)

  row = categories.iloc[0]

  category_colnames = row.apply(lambda x : x[:len(x) - 2])


  categories.columns = category_colnames
  categories.head()

  for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].astype(str).str[-1:]    
    # convert column from string to numeric
    categories[column] = categories[column].astype(int)

  df.drop(columns=['categories'], inplace=True)

  df = pd.concat([df, categories], axis=1)

  df.drop_duplicates(inplace=True)
  
  return df


def save_data(df, db_name):
  engine = create_engine('sqlite:///' + db_name)
  df.to_sql('messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print(type(df))#('Cleaning data...')
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
    
    
"""
def main():
  args = sys.argv[1:]
  df, categories = load_data(args[0], args[1])
  df = processing_data(df, categories)
  save_data(df, args[2])
"""
