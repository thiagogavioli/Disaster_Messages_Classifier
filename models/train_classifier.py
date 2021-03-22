#import the necessary libraries
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

import sys
import pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator,TransformerMixin

def load_data(database_filepath):
     """
    Load Data from the Database
    
    Arguments:
        database_filepath -> Path to SQLite database
    Output:
        X -> a dataframe containing features
        Y -> a dataframe containing labels
        category_names -> List of categories name
    """
    table = 'disaster_messages'
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql(table, engine)
    X = df['message']
    Y = df.iloc[:,4:]
    
    category_names = list(Y.columns.values)
    
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize the text
    
    Arguments:
        text -> Text message which needs to be tokenized
    Output:
        clean_tokens -> List of tokens extracted from text
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

# Build a custom transformer which will extract the starting verb of a sentence
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence and 
    create a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    """
    Build a model
    
    Output:
        An optimized Scikit ML Pipeline that process text messages and apply a classifier.
        
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            
            ('nlp_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer(use_idf=False))
            ])),
            
            ('sve', StartingVerbExtractor())
        ])),
        
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
        
        ])
    
    parameters = {'clf__estimator__n_estimators': [50, 100],
              'clf__estimator__learning_rate': [1,2]
    }
 
    return GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=3)


def evaluate_model(model, X_test, Y_test, category_names):
    """
    MultiOutput classification report
        
    Arguments:
        model -> model used to make the predictions
        X_test -> test features as input for the model
        y_test -> test target to compare with the new predictions
        category_names -> categories names from the target information
    
    Output:
        Classification report -> Report with precision, recall and F1-score
        for each category
    """
    Y_pred = model.predict(X_test)
    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], Y_pred[:,i]))
    return model


def save_model(model, model_filepath):
    """
    Save Pipeline model
    
    This function saves trained model as Pickle file, to be loaded later
    
    Arguments:
        model -> GridSearchCV or Scikit Pipeline object
        pickle_filepath -> destination path to save .pkl file
    
    """
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
     """
    Train Classifier Main function
    
    The function starts the Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) show the model performance on test set
        4) Save trained model as Pickle file
    
    """
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