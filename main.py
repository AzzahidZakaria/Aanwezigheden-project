import pandas as pd
import numpy as np
import joblib

def groepeer_activiteit(activiteit):
    activiteit_min = activiteit.lower()
    if 'examen' in activiteit_min:
        return 'Examen'
    elif activiteit == 'Activerend hoorcollege':
        return 'Activerend hoorcollege'
    elif activiteit == 'Oefensessie':
        return 'Oefensessie'
    elif activiteit == 'Practicum':
        return 'Practicum'
    else:
        return 'Andere'

def DataFrameTransformation(df):
    df = df.copy()
    df.columns = df.columns.str.replace('\ufeff', '').str.strip()
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df = df.drop("date", axis=1)
    df['until'] = np.where(df['until'] > 9999, df['until'] // 100, df['until'])
    df['from']  = np.where(df['from']  > 9999, df['from']  // 100, df['from'])
    from_decimal  = (df['from']  // 100) + ((df['from']  % 100) / 60)
    until_decimal = (df['until'] // 100) + ((df['until'] % 100) / 60)
    df['duration_hours'] = until_decimal - from_decimal
    df['start_hour'] = df['from'] // 100
    df['is_morning'] = (df['from'] < 1200).astype(int)
    df = df.drop(['from', 'until'], axis=1)
    df['activity'] = df['activity'].apply(groepeer_activiteit)
    df = df.drop("subgroup", axis=1)
    df['program'] = df['program'].str.split('-').str[:2].str.join('-')
    return df

df = pd.read_csv("lessen-subgroepen.csv", sep=';', decimal=',')

model = joblib.load('model.pkl')

df['aanwezigheidsgraad_predicted'] = model.predict(df).clip(0, 1).round(4)

print(df.to_string(index=False))