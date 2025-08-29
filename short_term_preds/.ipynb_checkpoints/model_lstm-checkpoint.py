import math
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow.keras as keras
from keras.optimizers import Adam
from scipy.special import inv_boxcox
from model_gp import get_next_n_weeks
from tensorflow.keras.activations import gelu 
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler

def schedule(epoch, lr):
    return lr * math.exp(-0.1)

def normalize_data(df, log_transform=False, ratio = 0.75, end_train_date = None):
    """
    Normalize features in the example table
    :param df:
    :param ratio: defines the size of the training dataset 
    :return:
    """
    
    if 'municipio_geocodigo' in df.columns:
        df.pop('municipio_geocodigo')

    for col in df.columns:
        if col.startswith('nivel'):
            # print(col)
            le = LabelEncoder()
            le.fit(df[col])
            df[col] = le.transform(df[col])

    df.fillna(0, inplace=True)
    if ratio != None:
        norm, norm_weights = normalize(df.iloc[:int(df.shape[0]*ratio)], norm='max', axis=0, return_norm = True)
        max_values = df.iloc[:int(df.shape[0]*ratio)].max()
    else:
        norm, norm_weights = normalize(df.loc[df.index <= f'{end_train_date}'], norm='max', axis=0, return_norm = True)
        max_values = df.loc[df.index <= f'{end_train_date}'].max()

    df_norm = df.divide(norm_weights, axis='columns')

    if log_transform==True:
        df_norm = np.log(df_norm)

    return df_norm, max_values

def create_df_for(ini_date_for, predict_n, df_pred, df_pred2_5, df_pred975):
    df = pd.DataFrame()

    for_dates = get_next_n_weeks(f'{ini_date_for}', predict_n)

    df['date'] = for_dates
    df['lower'] = df_pred2_5.iloc[-1].values
    df['pred'] = df_pred.iloc[-1].values
    df['upper'] = df_pred975.iloc[-1].values

    return df

def split_data_for(df, look_back=12, ratio=0.8, predict_n=5, Y_column=0, batch_size  = 4):
    """
    Split the data into training and test sets
    Keras expects the input tensor to have a shape of (nb_samples, timesteps, features).
    :param df: Pandas dataframe with the data.
    :param look_back: Number of weeks to look back before predicting
    :param ratio: fraction of total samples to use for training
    :param predict_n: number of weeks to predict
    :param Y_column: Column to predict
    :return:
    """

    s = get_next_n_weeks(ini_date=str(df.index[-1])[:10], next_days=predict_n)

    df = pd.concat([df, pd.DataFrame(index=s)])

    df = np.nan_to_num(df.values).astype("float64")
   
    n_ts = df.shape[0] - look_back - predict_n + 1
    # data = np.empty((n_ts, look_back + predict_n, df.shape[1]))
    data = np.empty((n_ts, look_back + predict_n, df.shape[1]))
    for i in range(n_ts):  # - predict_):
        #         print(i, df[i: look_back+i+predict_n,0])
        data[i, :, :] = df[i: look_back + i + predict_n, :]
     
    X_for = data[-1:, :look_back, ]

    return X_for

def get_nn_data_for(city, ini_date=None, end_date=None, look_back=4, predict_n=4, filename=None, batch_size = 4, 
                     end_train_date = '2024-04-21'):
    """
    :param city: int. The ibge code of the city, it's a seven number code 
    :param ini_date: string or None. Initial date to use when creating the train/test arrays 
    :param end_date: string or None. Last date to use when creating the train/test arrays
    :param end_train_date: string or None. Last day used to create the train data 
    :param ratio: float. If end_train_date is None, we use the ratio to spli the data into train and test 
    :param look_back: int. Number of last days used to make the forecast
    :param predict_n: int. Number of days forecast

    """
    df = pd.read_csv(filename, index_col='date')
    df.index = pd.to_datetime(df.index)

    try:
        target_col = list(df.columns).index(f"casos")
    except:
        target_col = list(df.columns).index(f"casos_est")
        
    df = df.dropna()

    if ini_date != None:
        df = df.loc[ini_date:]

    if end_date != None:
        df = df.loc[:end_date]

    norm_df, max_features = normalize_data(df, end_train_date = end_train_date)
    
    factor = max_features[target_col]

    X_for = split_data_for(
        norm_df,
        look_back=look_back,
        ratio=1,
        predict_n=predict_n,
        Y_column=target_col,
        batch_size=batch_size
    )

    return X_for, factor

def apply_forecast(city, ini_date, end_date, look_back, predict_n, filename, model_name, batch_size = 1,  end_train_date = '2023-10-01'):
    # (city, ini_date = None, end_date = None, look_back = 4, predict_n = 4, filename = filename
    X_for, factor = get_nn_data_for(city,
                                    ini_date=ini_date, end_date=end_date,
                                    look_back=look_back,
                                    predict_n=predict_n,
                                    filename=filename,  end_train_date = end_train_date
                                    )

    model = keras.models.load_model(f'./saved_models/{model_name}.keras', safe_mode=False, compile=False)

    pred = evaluate(model, X_for, batch_size=batch_size)

    pred = pred*factor 

    pred = inv_boxcox(pred, 0.05) - 1

    df_pred = pd.DataFrame(np.percentile(pred, 50, axis=2)) 
    df_pred2_5 = pd.DataFrame(np.percentile(pred, 2.5, axis=2)) 
    df_pred97_5 = pd.DataFrame(np.percentile(pred, 97.5, axis=2)) 

    df = create_df_for(end_date, predict_n, df_pred, df_pred2_5, df_pred97_5)

    df['step'] = [1,2,3]

    df['model'] = 'lstm'

    return df

def evaluate(model, Xdata, batch_size, uncertainty=True):
    """
    Function to make the predictions of the model trained 
    :param model: Trained lstm model 
    :param Xdata: Array with data
    :param uncertainty: boolean. If True multiples predictions are returned. Otherwise, just
                        one prediction is returned. 
    """
    if uncertainty:
        predicted = np.stack([model(Xdata, training =True) for i in range(100)], axis=2)
    else:
        predicted = model.predict(Xdata, batch_size=batch_size, verbose=0)
    return predicted


def build_lstm( hidden=8, features=100, predict_n=4, look_back=4, loss='msle', stateful = False, batch_size = 1,
                optimizer = Adam(learning_rate=0.001)):

    inp = keras.Input(
        #shape=(look_back, features),
        name='input', 
        batch_shape=(batch_size, look_back, features)
    )
    
    x = LSTM(
        hidden,
        input_shape=(look_back, features),
        #activation='relu',
        #recurrent_activation=gelu,
        stateful = stateful,
        name='lstm_1',
        
        return_sequences=True,
    )(inp, training=True)

    x = Dropout(0.2, name='dropout_1',)(x, training=True) 

    x = LSTM(
        hidden,
        input_shape=(look_back, features),
        #activation='relu',
        #recurrent_activation=gelu,
        stateful = stateful,
        name='lstm_2',
        
        return_sequences=True,
    )(x, training=True)

    x = Dropout(0.2, name='dropout_2',)(x, training=True) 

    x = LSTM(
        hidden,
        input_shape=(look_back, features),
        activation=gelu,
        #recurrent_activation=gelu,
        stateful = stateful,
        name='lstm_3',
        
        return_sequences=False,
    )(x, training=True)

    x = Dropout(0.2, name='dropout_3',)(x, training=True) 

    #x = BatchNormalization()(x, training = True)

    out = Dense(
        predict_n,
        activation='relu', 
        name='dense',
    )(x)
        #activity_regularizer=regularizers.L2(l2) )(x)
    model = keras.Model(inp, out)

    #optimizer = RMSprop(learning_rate=0.001, momentum= 0.5)
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy", "mape", "mse"])
    #print(model.summary())
    return model


def train(model, X_train, Y_train, label, batch_size=1, epochs=10, geocode=None, overwrite=True, cross_val=True,
          patience=20, monitor='val_loss', min_delta=0.00, verbose=0, doenca='dengue', save = True):
    """
    Train the lstm model 
    :param model: LSTM model compiled and created with the build_model function 
    :param X_train: array. Arrays with the features to train the model 
    :param Y_train: array. Arrays with the target to train the model
    :param label: string. Name to be used to save the model
    :param batch_size: int. batch size for batch training
    :param epochs: int.  Number of epochs used in the train 
    :param geocode: int. Analogous to city (IBGE code), it will be used in the name of the saved model
    :param overwrite: boolean. If true we overwrite a saved model with the same name. 
    :param validation_split: float. The slice of the training data that will be use to evaluate the model 
    """

    TB_callback = TensorBoard(
        log_dir="./tensorboard",
        histogram_freq=0,
        write_graph=True,
        write_images=True,
        update_freq='epoch',
        # embeddings_freq=10
    )

    seed = 7

    if cross_val == True:

            # Definição das camadas de validação cruzada
        kf = KFold(n_splits=4, shuffle=True, random_state=42)

        fold_no = 1
        for train_index, val_index in kf.split(X_train):
            
            print(f'Training fold {fold_no}...')

            # Split data
            X_train_, X_val_ = X_train[train_index], X_train[val_index]
            y_train_, y_val_ = Y_train[train_index], Y_train[val_index]

            hist = model.fit(
                        X_train_,
                        y_train_,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose,
                        validation_data=(X_val_, y_val_),
                        callbacks=[TB_callback, EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience)]
                    )
            
            fold_no = fold_no + 1
   
        if save: 
            model.save(f"saved_models/trained_{geocode}_{doenca}_{label}.keras", overwrite=overwrite)

    else:

        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.25,
                                                            random_state=seed)
        #print(X_train.shape)
        #print(X_test.shape)
        #print(model.summary())

        lr_scheduler = LearningRateScheduler(schedule)

        hist = model.fit(
            X_train,
            Y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, Y_test),
            verbose=verbose,
            callbacks=[TB_callback, EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience,
                                                  restore_best_weights=True),
                                                  lr_scheduler
                                                  ],
                                                  shuffle = False
        )

        if save: 

            model.save(f"saved_models/trained_{geocode}_{doenca}_{label}.keras", overwrite=overwrite)


    return model, hist


def train_model(model, city, doenca, ini_date=None, end_train_date=None,
                end_date=None, ratio=0.75, epochs=100,
                predict_n=4, look_back=4, batch_size=4,
                label='model', filename=None, verbose=0, patience = 50, cross_val = True, min_delta = 0.002):
    """
    The parameters ended with the word `date` are used to apply the model in different time periods. 
    :param model: tensorflow model. 
    :param city: int. IBGE code of the city. 
    :param doenca: string. Is used to name the trained model. 
    :param ratio: float. Percentage of the data used to train and test steps. 
    :param ini_date: string or None. Determines after which day the data will be used 
    :param end_train_date: string or None. Determines the last day used to train the data. 
                         If not None the parameter ratio is unused. 
    :param end_date: string or None. Determines the last day used 
    :param predict_n: int. Number of observations that it will be forecasted 
    :param look_back: int. Number of last observations used as input 
    :param label: string.
    :param filename: string. Path to the data used to train and evaluate the model. 
    """

    df, factor, X_train, Y_train, X_pred, Y_pred = get_nn_data(city, ini_date=ini_date,
                                                               end_date=end_date, end_train_date=end_train_date,
                                                               ratio=ratio, look_back=look_back,
                                                               predict_n=predict_n, filename=filename)

    model, hist = train(model, X_train, Y_train, label=label, batch_size=batch_size, epochs=epochs,
                                        geocode=city, overwrite=True, cross_val = cross_val, monitor='val_loss',
                                        verbose=verbose, doenca=doenca,
                                        min_delta = min_delta, patience=patience)

    return model, hist

def split_data(df, look_back=12, ratio=0.8, predict_n=5, Y_column=0):
    """
    Split the data into training and test sets
    Keras expects the input tensor to have a shape of (nb_samples, timesteps, features).
    :param df: Pandas dataframe with the data.
    :param look_back: Number of weeks to look back before predicting
    :param ratio: fraction of total samples to use for training
    :param predict_n: number of weeks to predict
    :param Y_column: Column to predict
    :return:
    """
    df = np.nan_to_num(df.values).astype("float64")
    # n_ts is the number of training samples also number of training sets
    # since windows have an overlap of n-1
    n_ts = df.shape[0] - look_back - predict_n + 1
    # data = np.empty((n_ts, look_back + predict_n, df.shape[1]))
    data = np.empty((n_ts, look_back + predict_n, df.shape[1]))
    for i in range(n_ts):  # - predict_):
        #         print(i, df[i: look_back+i+predict_n,0])
        data[i, :, :] = df[i: look_back + i + predict_n, :]
    # train_size = int(n_ts * ratio)
    train_size = int(df.shape[0] * ratio) - look_back
    #print(train_size)

    # We are predicting only column 0
    X_train = data[:train_size, :look_back, :]
    Y_train = data[:train_size, look_back:, Y_column]
    X_test = data[train_size:, :look_back, :]
    Y_test = data[train_size:, look_back:, Y_column]

    return X_train, Y_train, X_test, Y_test


def get_nn_data(city, ini_date = None, end_date = None, end_train_date = None, ratio = 0.75, look_back = 4, predict_n = 4, filename = None ):
    """
    :param city: int. The ibge code of the city, it's a seven number code 
    :param ini_date: string or None. Initial date to use when creating the train/test arrays 
    :param end_date: string or None. Last date to use when creating the train/test arrays
    :param end_train_date: string or None. Last day used to create the train data 
    :param ratio: float. If end_train_date is None, we use the ratio to spli the data into train and test 
    :param look_back: int. Number of last days used to make the forecast
    :param predict_n: int. Number of days forecast

    """
    df = pd.read_csv(filename, index_col = 'date' )
    df.index = pd.to_datetime(df.index)

    try:
        target_col = list(df.columns).index(f"casos")
    except ValueError:
        target_col = list(df.columns).index(f"casos_est")

    df = df.dropna()

    if ini_date != None: 
        df = df.loc[ini_date:]

    if end_date != None:
        df = df.loc[:end_date]

    if end_train_date == None: 
        
        norm_df, max_features = normalize_data(df, ratio = ratio)
        factor = max_features[target_col]

        X_train, Y_train, X_test, Y_test = split_data(
                norm_df,
                look_back= look_back,
                ratio=ratio,
                predict_n = predict_n, 
                Y_column=target_col,
        )
    
        # These variables will already concat the train and test array to easy the work of make 
        # the predicions of both 
        X_pred = np.concatenate((X_train, X_test), axis = 0)
        Y_pred = np.concatenate((Y_train, Y_test), axis = 0)

    else:
        norm_df, max_features = normalize_data(df, ratio = None, end_train_date = end_train_date)
        #print(norm_df.index[0])
        factor = max_features[target_col]

        # end_train_date needs to be lower than end_date, otherwise we will get an error in the value inside loc 
        if datetime.strptime(end_train_date, '%Y-%m-%d') < datetime.strptime(end_date, '%Y-%m-%d'):
            X_train, Y_train, X_test, Y_test = split_data(
                    norm_df.loc[norm_df.index <= end_train_date],
                    look_back= look_back,
                    ratio=1,
                    predict_n = predict_n, 
                    Y_column=target_col,
            )

            # X_pred and Y_pred will already concat the train and test array to easy the work of make 
            # the predicions of both 
            X_pred, Y_pred, X_test, Y_test = split_data(
                    norm_df,
                    look_back= look_back,
                    ratio=1,
                    predict_n = predict_n, 
                    Y_column=target_col,
            ) 

    return norm_df, factor,  X_train, Y_train, X_pred, Y_pred
