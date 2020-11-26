import pandas as pd
import preprocess

def scale_numeric(df, columns):
    df_numeric = preprocess.scale_data(df, columns)
    return df_numeric

def encode_categorical(df, columns_list):
    df_categorical = df[columns_list]
    for col in columns_list:
        df_categorical[col].astype('category')
    df_categorical = preprocess.ohe(df_categorical, df_categorical.columns)
    return df_categorical

def join_numeric_categorical(df_numeric, df_categorical):
    return df_numeric.join(df_categorical)