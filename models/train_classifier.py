# python train_classifier.py ../data/DisasterMessages.db my_model.pkl

# import packages
import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(data_file):
    """
    parameter data file label
    read in file
    load data from database
    clean data
    load to database
    define features and label arrays
    """
    engine = create_engine('sqlite:///' + data_file)
    df = pd.read_sql_table('messages', engine)
    
    df.dropna(inplace=True)

    # df.to_sql('messages', engine, index=False)

    X = df['message']
    y = df[df.columns.difference(['message', 'genre', 'id', 'original'])]

    return X, y


def tokenize(text):
    """
    tokenize, lemmatize and clean a text
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    
    
def build_model():
    """
    text processing and model pipeline
    define parameters for GridSearchCV
    create gridsearch object and return as final model pipeline
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenize)),
        ('tfidf', TfidfTransformer()),
        ('moc', MultiOutputClassifier(RandomForestClassifier(random_state=1)))
    ])
    
    pipeline.get_params().keys()
    parameters = {
        'moc__estimator__n_estimators': [50, 100, 200],
        'moc__estimator__min_samples_split': [2, 3, 4]
    }
    
    model_pipeline = GridSearchCV(pipeline, param_grid=parameters)

    return model_pipeline


def train(X, y, model):
    """
    train test split
    fit model
    output model test results
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, index = y_test.index, columns = y_test.columns)
    for column in y_train.columns:
        print(column)
        print(classification_report(y_test[column], y_pred_df[column]))

    return model


def export_model(model, model_path):
    """
    Export model as a pickle file
    """
    pickle.dump(model, open(model_path, 'wb'))


def run_pipeline(data_file, model_path):
    """
    run ETL pipeline
    build model pipeline
    train model pipeline
    save model
    """
    X, y = load_data(data_file)  
    model = build_model()  
    model = train(X, y, model)  
    export_model(model, model_path)  

if __name__ == '__main__':
    data_file = sys.argv[1]  # get filename of dataset
    model_path = sys.argv[2]  # get model path
    run_pipeline(data_file, model_path)  # run data pipeline
    


