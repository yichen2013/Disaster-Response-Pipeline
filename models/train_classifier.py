import sys
from sqlalchemy import create_engine
import pandas as pd

import string
import operator

# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet','stopwords','averaged_perceptron_tagger'])

import re
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.metrics import classification_report

from sklearn.model_selection import RandomizedSearchCV

from sklearn.base import BaseEstimator, TransformerMixin

def load_data(database_filepath):
    '''
    Load data function
    
    '''
    engine = create_engine('sqlite:///'+ database_filepath)
    
    df = pd.read_sql("Messages", engine)

    X = df['message']
    
    #get Y. child_alone is all 0, drop it.
    Y = df.drop(['id', 'message', 'original', 'genre','child_alone'], axis = 1)
    
    #change the response 2 in related to 1, since I only need to know that it happened.
    Y['related']=Y['related'].apply(lambda x: 1 if x == 2 else x)
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    '''The function will tokenize the text
    return the cleaned text'''
    #Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
     # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer# lemmatize, normalize case, and remove leading/trailing white space
    cleaned_tokens = [WordNetLemmatizer().lemmatize(w.strip()) for w in tokens]
    
    return cleaned_tokens


def build_model():
    #Build pipeline
    model = pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
      
    ])
    
    '''parameters ={'vect__min_df': [1, 5],
              'tfidf__use_idf':[True, False],
              'clf__estimator__n_estimators':[10, 20, 25,30 ], 
              'clf__estimator__min_samples_split':[2, 5, 10]
              }
    cv = RandomizedSearchCV(pipeline, param_distributions=parameters,random_state=0)'''
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model function
    Arguments:
    model: Scikit ML Pipeline
    X_test: test features
    Y_test: test labels
    category_names: label names(multi output'''
    Y_pred = model.predict(X_test)
    
    print(classification_report(Y_test.values, Y_pred,target_names=category_names))
    results = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])
def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


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