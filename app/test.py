from sqlalchemy import create_engine
import pandas as pd

import sys
import string
import operator
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
def main():
    print(len(sys.argv),sys.argv[1:])
    if len(sys.argv) == 2:
        database_filepath= sys.argv[1]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath) 
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')
if __name__ == '__main__':
    main()        