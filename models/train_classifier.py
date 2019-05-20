import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import sqlite3
from sqlalchemy import create_engine

import re
import numpy as np
import pandas as pd
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):
    '''
    loads data from SQL database into dataframe. Dataframe is split in two. Names of Categories are extracted for use below.
    input:
        path to SQL database
    output:
        messages
        categories of messages
        names of the categories
    '''
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_query('select * from cleandata', engine)
    target_cols = df.columns[4:].tolist()
    X = df.message.values
    Y = df[target_cols].values
    
    return X, Y, target_cols


def tokenize(text):
    '''
    ingests a message and tokenizes it for count vectorizer
    input:
        messages
    output:
        clean tokens
    '''
     # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()
    
    # lemmatize andremove stop words
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Builds model using pipeline. Random Forest Classifier using 100 Estimators and min 3 samples split. These were the optimal parameters indentified in GridSearchCV.
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('moc',MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    return pipeline
    


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    uses model to predict categories of test population and compares predictions to actuals to get model score.
    input:
        model
        test population of messages
        test population of categories
        names of categories
    output:
        Prints F1score, precision and recall for each column and overall accuracy of model for test group
    '''
    y_preds = model.predict(X_test)
    df_y_test = pd.DataFrame(Y_test)
    df_y_test.columns = category_names
    df_y_pred = pd.DataFrame(y_preds)
    df_y_pred.columns = category_names
    
    #calculate overall accuracy for all predicted columns
    overall_accuracy = (y_preds == Y_test).mean().mean()
    
    print('Average overall accuracy {0:.2f}% \n'.format(overall_accuracy*100))
    
    #Calculate F1score, precision and recall for each column
    for col in category_names:
        print(col,'\n')
        print(classification_report(np.array(df_y_test[col]),np.array(df_y_pred[col])))
        
    return None


def save_model(model, model_filepath):
    '''
    takes trained model and saves to filepath for future use.
    input:
        model
        filepath for saved model
    output:
        None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))
    
    return None


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()