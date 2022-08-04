# process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages, categories):
  # load messages dataset
  messages = pd.read_csv(messages)
  # load categories dataset
  categories = pd.read_csv(categories.css)

  # merge datasets
  df = pd.merge(messages, categories)
  
  return df


def preprocessing_data(df):
  # create a dataframe of the 36 individual category columns
  categories = categories['categories'].str.split(';', expand=True)

  # select the first row of the categories dataframe
  row = categories.iloc[0]

  # use this row to extract a list of new column names for categories.
  # one way is to apply a lambda function that takes everything 
  # up to the second to last character of each string with slicing
  category_colnames = row.apply(lambda x : x[:len(x) - 2])

  # rename the columns of `categories`
  categories.columns = category_colnames
  categories.head()

  for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].astype(str).str[-1:]    
    # convert column from string to numeric
    categories[column] = categories[column].astype(int)

  # drop the original categories column from `df`
  df.drop(columns=['categories'], inplace=True)

  # concatenate the original dataframe with the new `categories` dataframe
  df = pd.concat([df, categories], axis=1)

  # drop duplicates
  df.drop_duplicates(inplace=True)


def save_data(df, db_name):
  engine = create_engine('sqlite:///' + db_name)
  df.to_sql('messages', engine, index=False)


def main():
  args = sys.argv[1:]
  df = load_data(args[0], args[1])
  df = preprocessing_data(df)
  save_data(df, args[2])


main()
