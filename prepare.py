import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def prep_iris(iris):
    iris = iris.drop(columns=['species_id', 'measurement_id'])
    iris = iris.rename(columns={'species_name':'species'})
    iris_dummy = pd.get_dummies(iris.species, drop_first=True)
    iris = pd.concat([iris, iris_dummy], axis=1)
    return iris

def prep_titanic(titanic):
    titanic = titanic.set_index('passenger_id').drop(columns=['class', 'embark_town', 'age', 'deck'])
    titanic_dummy = pd.get_dummies(titanic[['sex', 'embarked']], drop_first=[True,True])
    titanic = pd.concat([titanic, titanic_dummy], axis=1)
    return titanic

def prep_telco(telco):
    telco = telco.set_index('customer_id').drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'paperless_billing'])
    telco = telco.replace(['Yes', 'No'], [1,0])
    telco_dummies = pd.get_dummies(telco[['gender','multiple_lines', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'contract_type', 'internet_service_type', 'payment_type']], drop_first=True)
    telco = pd.concat([telco, telco_dummies], axis=1)
    telco.total_charges = telco.total_charges.replace(' ', 0).astype(float)
    return telco

def split_data(df, target):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on survived.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[target])
    train, validate = train_test_split(train_validate, 
                                       test_size=.25, 
                                       random_state=123, 
                                       stratify=train_validate[target])
    return train, validate, test