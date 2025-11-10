import pandas as pd


def load_data(path):
    return pd.read_csv(path)


def preprocess(data):
    data = data.dropna()
    X = data.drop(columns=['target'])
    y = data['target']
    return X, y
