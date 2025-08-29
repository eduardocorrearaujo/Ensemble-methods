import gpflow
import numpy as np
import pandas as pd 
import tensorflow as tf 
from epiweeks import Week
from scipy.special import inv_boxcox
from datetime import datetime, timedelta
from gpflow.mean_functions import Constant
from sklearn.preprocessing import StandardScaler

class ExportableGPModel(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 21], dtype=tf.float64)])
    def predict(self, Xnew):
        mean, var = self.model.predict_y(Xnew)
        return mean, var

def build_lagged_features(dt, lag=2, dropna=True):
    '''
    returns a new DataFrame to facilitate regressing over all lagged features.
    :param dt: Dataframe containing features
    :param lag: maximum lags to compute
    :param dropna: if true the initial rows containing NANs due to lagging will be dropped
    :return: Dataframe
    '''
    if type(dt) is pd.DataFrame:
        new_dict = {}
        for col_name in dt:
            new_dict[col_name] = dt[col_name]
            # create lagged Series
            for l in range(1, lag + 1):
                new_dict['%s_lag%d' % (col_name, l)] = dt[col_name].shift(l)
        res = pd.DataFrame(new_dict, index=dt.index)

    elif type(dt) is pd.Series:
        the_range = range(lag + 1)
        res = pd.concat([dt.shift(-i) for i in the_range], axis=1)
        res.columns = ['lag_%d' % i for i in the_range]
    else:
        print('Only works for DataFrame or Series')
        return None
    if dropna:
        return res.dropna()
    else:
        return res


def gp_model(X_train_, y_train_, idx_time): 
    
    k1 = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(active_dims = [idx_time])) # kernel temporal 
    k2 = gpflow.kernels.Matern32(X_train_.shape[1]) # todas as curvas 

    kernel = k1 + k2 

    m = gpflow.models.GPR(data =(X_train_, y_train_),kernel =  kernel,
                         mean_function=Constant(np.mean(y_train_)) )
    
    opt = gpflow.optimizers.Scipy()
    opt.minimize(m.training_loss, m.trainable_variables)
    
    return m
    
def get_next_n_weeks(ini_date: str, next_days: int) -> list:
    """
    Return a list of dates with the {next_days} days after ini_date.
    This function was designed to generate the dates of the forecast
    models.
    Parameters
    ----------
    ini_date : str
        Initial date.
    next_days : int
        Number of days to be included in the list after the date in
        ini_date.
    Returns
    -------
    list
        A list with the dates computed.
    """

    next_dates = []

    a = datetime.strptime(ini_date, "%Y-%m-%d")

    for i in np.arange(1, next_days + 1):
        d_i = datetime.strftime(a + timedelta(days=int(i * 7)), "%Y-%m-%d")

        next_dates.append(datetime.strptime(d_i, "%Y-%m-%d").date())

    return next_dates

def preencher_nan_com_anterior_mais_um(serie):
    # Percorre a coluna e substitui NaN pelo valor anterior + 1
    for i in range(1, len(serie)):
        if pd.isna(serie.iloc[i]):
            serie.iloc[i] = serie.iloc[i-1] + 1
    return serie

def norm_data(X_train, y_train, X_test, y_test): 
    
    sc_x = StandardScaler()
    sc_y = StandardScaler()

    sc_x.fit(X_train)
    sc_y.fit(y_train.values.reshape(-1, 1))

    X_train_ = sc_x.transform(X_train)
    #y_train_ = sc_y.transform(y_train.values.reshape(-1, 1))
    #y_train_ = y_train.values.reshape(-1, 1)
    X_test_ = sc_x.transform(X_test)
    #y_test_ = sc_y.transform(y_test.values.reshape(-1, 1))
    #y_test_ = y_test.values.reshape(-1, 1)
    
    
    return X_train_,  X_test_
    
def preprocess_data(data, d =10, look_back = 11, 
                    ini_train = '2015-06-01',
                    end_train = '2022-08-21',
                    ini_test = '2022-08-21',
                    end_test = '2023-08-21', dropna = True): 
    
    data = data.fillna(0)
    data = data.loc['2010-01-01':]

    data.replace([np.inf, -np.inf], 0, inplace=True)
    
    data_lag = build_lagged_features(data, look_back, dropna=dropna)

    data_lag['SE_adj'] = data_lag['SE'].astype(str).str[-2:].astype(int).shift(-d)

    data_lag.drop('SE', axis = 1, inplace = True)
    target = data_lag['casos'].shift(-d)
    target = target.dropna()   
    
    #data_lag = data_lag.dropna()

    cols = []

    for l in np.arange(1, look_back+1):
        cols.append(f'SE_lag{l}')
    
    data_lag = data_lag.drop(cols, axis=1)

    #data_lag['SE_adj'] = data_lag['SE_adj'].isna() + data_lag['SE_adj'].fillna(method='ffill') 
    
    # Aplica a função à coluna
    data_lag['SE_adj'] = preencher_nan_com_anterior_mais_um(data_lag['SE_adj'])

    X_train = data_lag.loc[(data_lag.index >= ini_train) & (data_lag.index < end_train)]

    y_train = target.loc[(target.index >= ini_train) & (target.index < end_train)]

    X_test = data_lag.loc[(data_lag.index >= ini_test) & (data_lag.index <= end_test)]

    y_test = target.loc[(target.index >= ini_test) & (target.index <= end_test)]


    X_train_,  X_test_ = norm_data(X_train, y_train, X_test, y_test)
    
    return data_lag, target, X_train_, y_train, X_test_, y_test
    

def train_model(state, end_train, ini_train = '2015-06-01'):
    '''
    Function to train and save the gp model 
    '''

    df_ = pd.read_csv(f'data/dengue_{state}.csv.gz', index_col = 'date')

    df_.index = pd.to_datetime(df_.index)

    data_lag, target, X_train, y_train, X_test, y_test = preprocess_data(df_, 3, 3, 
                                                                        ini_train = ini_train,
                                                                        end_train = end_train,
                                                                        ini_test = end_train,
                                                                        end_test = end_train)

    m = gp_model(X_train, y_train.values.reshape(-1,1), idx_time=20)

    exportable_model = ExportableGPModel(m)
    tf.saved_model.save(exportable_model, f"saved_models/gp_{state}")
    
    return 

def apply_model(state, end_date):
    '''
    Function to load and apply the gp model 
    '''

    df_ = pd.read_csv(f'data/dengue_{state}.csv.gz', index_col = 'date')

    df_.index = pd.to_datetime(df_.index)

    end_train = (pd.to_datetime(end_date) - timedelta(weeks=2)).strftime('%Y-%m-%d')

    data_lag, target, X_train, y_train, X_test, y_test = preprocess_data(df_, 3, 3, 
                                                                        ini_train = '2015-06-01',
                                                                        end_train = end_train,
                                                                        ini_test = end_train,
                                                                        end_test = end_date)
    
    for_week = Week.fromdate(pd.to_datetime(end_date)).week
    
    m = tf.saved_model.load(f"saved_models/gp_{state}")

    pred_te, var_te  = m.predict(X_test)

    df_preds = pd.DataFrame()

    df_preds['date'] = get_next_n_weeks(end_date, 3)

    df_preds['pred'] = inv_boxcox(pred_te, 0.05) -1

    df_preds['lower'] = inv_boxcox(pred_te - 1.96*var_te, 0.05).reshape(1,-1)[0] -1 
    
    df_preds['upper'] = inv_boxcox(pred_te + 1.96*var_te, 0.05).reshape(1,-1)[0]  -1 

    df_preds['step'] = [1,2,3]

    df_preds['model'] = 'gp'

    #df_preds.to_csv(f'forecast_tables/for_gp_se_{for_week}_{state}.csv.gz', index = False)
    
    return df_preds 