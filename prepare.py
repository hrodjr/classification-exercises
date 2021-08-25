def clean_data(df):

#drop duplicates   
    df = df.drop_duplicates()
#replaces empty values with 0
    df = df.replace({'total_charges': ' '}, 0)
#drop redundant columns
    df = df.drop(['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id'], axis = 1)
#change from object to float
    df['total_charges'] = df['total_charges'].astype(float)
#Replaces 'no ... service' with 'No' for encoding.
    df = df.replace(to_replace = 'No internet service', value = 'No')
    df = df.replace(to_replace = 'No phone service', value = 'No')

#encode by creating a dummy df.
    dummy_df = pd.get_dummies(df[['dependents','phone_service','online_security','online_backup','payment_type','internet_service_type','contract_type','gender','partner','multiple_lines','device_protection','tech_support','streaming_tv','streaming_movies','paperless_billing','churn']], dummy_na=False, drop_first=[True, True])
#drop and replace columns with dummy df. 
    df = df.drop(columns=['dependents','phone_service','online_security','online_backup','payment_type','internet_service_type','contract_type','gender','partner','multiple_lines','device_protection','tech_support','streaming_tv','streaming_movies','paperless_billing','churn'])
    df = pd.concat([df, dummy_df], axis=1)

return df

############################### Split Data ##################################

def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on survived.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.survived)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123, stratify=train_validate.survived)
    
return train, validate, test