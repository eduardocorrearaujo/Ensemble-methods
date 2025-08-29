import numpy as np
import pandas as pd
from epiweeks import Week
from scipy.stats import boxcox

states_BR = ['AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI',
 'SE', 'RN', 'SP', 'MG', 'RJ', 'ES', 'AM', 'AP', 'TO',
 'RR', 'RO', 'AC', 'PA', 'DF', 'GO', 'MT', 'MS',
 'RS', 'SC', 'PR']

def filter_agg_data(state):
    '''
    Get the state data and aggregate it by week
    '''
    df = pd.read_parquet(f'../load_infodengue_data/data/cases/{state}_dengue.parquet', columns=['municipio_geocodigo', 'casos_est', 'casprov'])

    df.index = pd.to_datetime(df.index) 
    
    df = df.loc[df.index >= '2023-06-01']
    
    df = df.resample('W-SUN').sum().drop(['municipio_geocodigo'], axis =1).reset_index()

    df['uf'] = state

    return df 

def up_data(df_up, prop_state, state): 
    '''
    função para atualizar os dados a partir da proporção observada
    '''
    df_up_ = df_up.loc[df_up.uf == state].copy()

    df_up_.loc[:, 'casos'] = prop_state[state]*df_up_['casos_est']

    df_up_ = df_up_.drop(['casos_est', 'casprov'],axis =1)
    
    df_up_ = df_up_[['date','casos', 'uf']]

    return df_up_

# funções para criar as features utilizadas para treinar os modelos de forecast
def calcular_metricas_por_janela(array, tamanho_janela, funcoes):
    # Criar um array com as janelas deslizantes
    janelas = np.lib.stride_tricks.sliding_window_view(array, tamanho_janela)

    # Aplicar as funções de interesse em cada janela
    resultados = [func(janela, axis=0) for func in funcoes for janela in janelas]
    
    return np.array(resultados)

def get_slope(casos, axis = 0): 
     
    return np.polyfit(np.arange(0,4), casos, 1)[0]

def org_data(df_all, state):
    if state == 'BR':
        df = df_all.copy()
    else: 
        df = df_all.loc[df_all.uf == state].copy()

    df.date = pd.to_datetime(df.date)
    
    df.set_index('date', inplace = True)

    df = df[['casos']].resample('W-SUN').sum()

    df['casos'] = boxcox(df.casos+1, lmbda=0.05)

    df['SE'] = [Week.fromdate(x) for x in df.index]

    df = df[['SE', 'casos']].sort_index()
    
    df['SE'] = df['SE'].astype(str).str[-2:].astype(int)
    
    df['SE'] = df['SE'].replace(53,52)
    
    df['diff_casos'] = np.concatenate( (np.array([np.nan]), np.diff(df['casos'], 1)))
    
    array = np.array(df.casos)
    tamanho_janela = 4
    
    df['casos_mean'] =  np.concatenate( (np.array([np.nan, np.nan, np.nan]), calcular_metricas_por_janela(array, tamanho_janela, [np.mean])))
    
    df['casos_std'] =  np.concatenate( (np.array([np.nan, np.nan, np.nan]), calcular_metricas_por_janela(array, tamanho_janela, [np.std])))
    
    df['casos_slope'] =  np.concatenate( (np.array([np.nan, np.nan, np.nan]), calcular_metricas_por_janela(array, tamanho_janela, [get_slope])))
    
    df = df.dropna()

    df.to_csv(f'data/dengue_{state}.csv.gz')


if __name__ == '__main__':
    
    df_prop = pd.DataFrame()

    for state in states_BR: 
        df_prop = pd.concat([df_prop, filter_agg_data(state)])

    # definindo o intervalo dos últimos 6 meses para o cálculo das proporções
    end_date_prop = df_prop.data_iniSE.max() - pd.DateOffset(weeks=10)
    begin_date_prop = df_prop.data_iniSE.max() - pd.DateOffset(weeks=34)

    print(begin_date_prop)
    print(end_date_prop)

    # dicionário onde serão salvas as proporções 
    prop_state = dict()

    for state in states_BR: 
        df_filter = df_prop.loc[(df_prop.uf == state) & (df_prop.data_iniSE >= begin_date_prop) & (df_prop.data_iniSE <= end_date_prop)]

        prop_state[state]= np.mean(df_filter['casprov']/df_filter['casos_est'])


    ### dados de antes de 2024
    df_atual = pd.read_csv('data/dengue_up.csv.gz', index_col = 'date', usecols = ['date', 'regional_geocode', 'casos', 'uf'])

    df_atual.index = pd.to_datetime(df_atual.index)

    df_atual['uf'] = df_atual['uf'].str[:2]

    df_atual = df_atual.groupby('uf').resample('W-SUN').sum().drop(['uf', 'regional_geocode'], axis = 1).reset_index()

    #df_atual = df_atual.loc[df_atual.date <= end_date_prop]
    df_atual = df_atual.loc[df_atual.date < '2024-01-01']

    ## renomeando as colunas do novo dataset para conseguir concatenar com o antigo  
    df_prop.data_iniSE = pd.to_datetime(df_prop.data_iniSE)

    df_prop.set_index('data_iniSE', inplace = True)

    df_up_no_delay =  df_prop.loc[(df_prop.index <= end_date_prop) & (df_prop.index >= '2024-01-01')].rename(columns = {'casprov': 'casos'}).drop('casos_est',axis =1)

    df_up_no_delay = df_up_no_delay.reset_index().rename(columns= {'data_iniSE':'date'})[['date', 'casos', 'uf']]
    df_up_cor = df_prop.loc[df_prop.index > end_date_prop].reset_index().rename(columns= {'data_iniSE':'date'})

    # dados atualizados 
    #df_update = df_atual
    df_update = pd.concat([df_atual, df_up_no_delay])
    
    for state in states_BR:

        df_update  = pd.concat([df_update, up_data(df_up_cor, prop_state, state)])

    df_update.to_csv('data/dengue_update.csv.gz')

    for state in states_BR:
        org_data(df_update, state)

    org_data(df_update, 'BR')