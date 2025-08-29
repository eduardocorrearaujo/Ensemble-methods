import numpy as np
import pandas as pd 
import model_gp as gp
import model_arima as ar 
import model_lstm as lstm 
from epiweeks import Week

states_BR = ['AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI',
 'SE', 'RN', 'SP', 'MG', 'RJ', 'ES', 'AM', 'AP', 'TO',
 'RR', 'RO', 'AC', 'PA', 'DF', 'GO', 'MT', 'MS',
 'RS', 'SC', 'PR', 'BR']

df = pd.read_csv(f'data/dengue_BR.csv.gz')
df.date = pd.to_datetime(df.date)
end_date = df.date.max().strftime('%Y-%m-%d')
for_week = Week.fromdate(df.date.max()).week

for state in states_BR:
    try: 
        #print('--------------------- Apply ARIMA ---------------------')

        df = pd.read_csv('data/dengue_update.csv.gz', index_col = 'Unnamed: 0')

        df.date = pd.to_datetime(df.date)

        df = df.loc[df.date <= end_date]

        df.set_index('date', inplace = True)

        df = df.groupby('uf').resample('W-SUN').sum().drop(['uf'], axis = 1).reset_index()
        #df = df.loc[df.uf == 'MG']

        df = df.rename(columns = {'date': 'dates',
                                'casos': 'y'})
        
        ar.apply_model(df, state, for_week)

        #print('--------------------- Apply GP ---------------------')

        gp.apply_model(state, end_date)
        
        #print('--------------------- Apply LSTM ---------------------')

        FILENAME_DATA = f'data/dengue_{state}.csv.gz'
        df_ = pd.read_csv(FILENAME_DATA, index_col = 'date')

        feat = df_.shape[1]
        
        model_name = f'trained_{state}_dengue_state'

        print(model_name)

        df_for = lstm.apply_forecast(state, None, end_date, look_back=4, predict_n=3,
                                        filename=FILENAME_DATA, model_name=model_name)

        df_for.date = pd.to_datetime(df_for.date)

        df_for.to_csv(f'forecast_tables/for_lstm_se_{for_week}_{state}.csv.gz', index = False)

    except: 
        print(state)
        pass