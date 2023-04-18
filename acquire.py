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