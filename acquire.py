import pandas as pd
import os
from env import get_db_url 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

link = os.getcwd()+'/'

def get_titanic_data():
    url = get_db_url('titanic_db')
    SQL_query = 'SELECT * FROM passengers'
    if os.path.exists(link + 'titanic.csv'):
        df = pd.read_csv('titanic.csv')
        return df
    else:
        df = pd.read_sql(SQL_query,url)
        df.to_csv('titanic.csv', index=False)
        return df
    

def get_iris_data():
    url = get_db_url('iris_db')
    SQL_query = '''SELECT * FROM measurements
	            JOIN species USING(species_id);'''
    if os.path.exists(link + 'iris.csv'):
        df = pd.read_csv('iris.csv')
        return df
    else:
        df = pd.read_sql(SQL_query,url)
        df.to_csv('iris.csv', index=False)
        return df

def get_telco_data():
    url = get_db_url('telco_churn')
    SQL_query = '''SELECT * FROM customers
	            JOIN contract_types USING(contract_type_id)
                JOIN internet_service_types USING(internet_service_type_id)
                JOIN payment_types USING(payment_type_id);'''
    if os.path.exists(link + 'telco.csv'):
        df = pd.read_csv('telco.csv')
        return df
    else:
        df = pd.read_sql(SQL_query,url)
        df.to_csv('telco.csv', index=False)
        return df

def prep_iris(iris):
    iris = iris.drop(columns=['species_id', 'measurement_id'])
    iris = iris.rename(columns={'species_name':'species'})
    iris_dummy = pd.get_dummies(iris.species, drop_first=True)
    iris = pd.concat([iris, iris_dummy], axis=1)
    return iris

def prep_titanic(titanic):
    titanic = titanic.set_index('passenger_id').drop(columns=['class', 'embark_town', 'deck'])
    titanic_dummy = pd.get_dummies(titanic[['sex', 'embarked']], drop_first=[True,True])
    titanic = pd.concat([titanic, titanic_dummy], axis=1)
    return titanic

def prep_telco(telco):
    telco = telco.set_index('customer_id').drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'paperless_billing'])
    telco = telco.replace(['Yes', 'No'], [1,0])
    telco_dummies = pd.get_dummies(telco[['gender','multiple_lines', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'contract_type', 'internet_service_type', 'payment_type']], drop_first=True)
    telco = pd.concat([telco, telco_dummies], axis=1)
    return telco

def split_data(df, target):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on survived.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[target])
    train, validate = train_test_split(train_validate, 
                                       test_size=.2, 
                                       random_state=123, 
                                       stratify=train_validate[target])
    return train, validate, test