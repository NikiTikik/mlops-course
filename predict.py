# predict.py

import pandas as pd

FEATURE_ORDER = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']


def predicted_prices(model, features_list):
    """
    Делает предсказания стоимости поездок на основе входных признаков.
    
    Parameters:
        model (GradientBoostingRegressor): Обученная модель.
    
    Returns:
        list: Список предсказанных значений стоимости поездок.
    """
    df = pd.DataFrame(features_list, columns=FEATURE_ORDER)
    predictions = model.predict(df)
    return predictions.tolist()