import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


def clean_data(df):

#drop duplicates   
    df = df.drop_duplicates()
#replaces empty values with 0
    df = df.replace({'total_charges': ' '}, 0)
#drop redundant columns
    df = df.drop(['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id','senior_citizen','paperless_billing'], axis = 1)
#change from object to float
    df['total_charges'] = df['total_charges'].astype(float)
#Replaces 'no ... service' with 'No' for encoding.
    df = df.replace(to_replace = 'No internet service', value = 'No')
    df = df.replace(to_replace = 'No phone service', value = 'No')

#encode by creating a dummy df.
#get_dummies creates a seperate df of booleans for the identified columns below. Cleaning for the decission tree.
    dummy_df = pd.get_dummies(df[['churn',
    'gender',
    'partner',
    'dependents',
    'phone_service',
    'multiple_lines',
    'online_security',
    'online_backup',
    'device_protection',
    'tech_support',
    'streaming_tv',
    'streaming_movies']], dummy_na=False, drop_first=[True, True])

#set 'drop_first' to 'False' to encode multiple types of the below listed columns.
    dummy_df_types = pd.get_dummies(df[['payment_type','internet_service_type','contract_type',]], dummy_na=False, drop_first=False)

#now drop the above two columns...
    df = df.drop(columns=['dependents',
    'phone_service',
    'online_security',
    'online_backup',
    'payment_type',
    'internet_service_type',
    'contract_type',
    'gender',
    'partner',
    'multiple_lines',
    'device_protection',
    'tech_support',
    'streaming_tv',
    'streaming_movies',
    'churn'])

#...and concatanate the dummies df with the prep's df.
    df = pd.concat([df, dummy_df_types, dummy_df], axis=1)

#rename columns
    df = df.rename(columns={'payment_type_Bank transfer (automatic)':'bank_transfer',
    'payment_type_Credit card (automatic)':'credit_card',
    'payment_type_Electronic check':'e_check',
    'payment_type_Mailed check':'check',
    'internet_service_type_DSL':'dsl',
    'internet_service_type_Fiber optic':'fiber',
    'internet_service_type_None':'no_internet',
    'contract_type_Month-to-month':'m2m',
    'contract_type_One year':'one_year_contract',
    'contract_type_Two year':'two_year_contract',
    'dependents_Yes':'dependents',
    'phone_service_Yes':'phone_service',
    'online_security_Yes':'online_security',
    'online_backup_Yes':'online_backup',
    'gender_Male':'is_male',
    'partner_Yes':'partner',
    'multiple_lines_Yes':'multiple_lines',
    'device_protection_Yes':'device_protection',
    'tech_support_Yes':'tech_support',
    'streaming_tv_Yes':'streaming_tv',
    'streaming_movies_Yes':'streaming_movies',
    'paperless_billing_Yes':'paperless',
    'churn_Yes':'churned'})

#after encoding the decision to concatanate bank_transfer and credit_card into auto_payment was made to whether or not autopayments had a play on churn.    
#combine 'bank_transfer' and 'credit_card' with new 'auto_payment' column
    df['autopayment'] = df['bank_transfer'] + df['credit_card']
    df['not_autopayment'] = df['e_check'] + df['check']

#drop 'bank_transfer' and 'credit_card' columns
    df = df.drop(columns=['bank_transfer', 'credit_card', 'e_check', 'check'])

#dropped the following columns after running univariate and bivariate statistics
    df = df.drop(columns=['is_male', 'online_security', 'online_backup', 'device_protection', 'one_year_contract', 'two_year_contract', 'multiple_lines', 'total_charges'])

    return df


############################### Split Data ##################################

def train_validate_test_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    
    train_validate, test = train_test_split(df, test_size=0.2, random_state=seed, stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=seed, stratify=train_validate[target])
    return train, validate, test