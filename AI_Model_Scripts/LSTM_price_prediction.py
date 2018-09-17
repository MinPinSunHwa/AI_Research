import numpy as np
import tensorflow as tf
import time
import pandas as pd
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as K
from pymongo import *
from datetime import time, tzinfo, timedelta, datetime
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM 
from keras.models import Sequential 
from keras.layers import Dense 
import keras.backend as K 
from keras.callbacks import EarlyStopping

def get_data():
    client = MongoClient('52.79.239.183', 27017)
    print("DB connection complete!!")
    DB_Coin = client["BINANCE"]
    Collection = DB_Coin['BTC/USDT_30MIN']
    DB_schema = []
    
    for collect in Collection.find():    
        temp_record = {}
        temp_record['open'] = collect['price_open']
        temp_record['close'] = collect['price_close']
        temp_record['high'] = collect['price_high']
        temp_record['low'] = collect['price_low']
        temp_record['volume'] = collect['volume_traded']    
        temp_record['Date'] = collect['time_close']
        DB_schema.append(temp_record)
        del temp_record
     
    return DB_schema




# 30min *4 = 2 hours 이므로 4 unit time 만큼 shift해 준다

def Shift_DF(DF):
    for s in range(1,5):
        DF['close_shift_{}'.format(s)] = DF['close'].shift(s)        
    return DF



def get_model(x_train):
    model = Sequential() # Sequeatial Model 
    model.add(LSTM(50, input_shape=(x_train.shape[1], 1))) # (timestep, feature) 
    model.add(Dense(1)) # output = 1 
    model.compile(loss='mean_squared_error', optimizer='adam') 
    model.summary()
    
    early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
    model.fit(x_train, y_train, epochs=100, batch_size=30, verbose=1, callbacks=[early_stop])
    return model



def plot_comparison(model, start_idx, length=100, train=True):
    """
    Plot the predicted and true output-signals.
    :param start_idx: Start-index for the time-series.
    :param length: Sequence-length to process and plot.
    :param train: Boolean whether to use training- or test-set.
    """
    if train:
        # Use training-data.
        x = x_train
        y_true = y_train
        loss_name = "train"
    else:
        # Use test-data.
        x = x_test
        y_true = y_test
        loss_name = "test"    
    # End-index for the sequences.
    end_idx = start_idx + length
    # Select the sequences from the given start-index and
    # of the given length.
    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]
    # Input-signals for the model.
    #x = np.expand_dims(x, axis=0)
    y_pred = model.predict(x)
    test_score = model.evaluate(x, y_true, verbose=0)
    print("%s loss : %.6f" %(loss_name, test_score))
    
    # Min-Max Scaling 했던 값들을 다시 원래 스케일로 복구시킨다
    y_pred_rescaled = sc.inverse_transform(y_pred)
    y_true_rescaled = sc.inverse_transform(y_true)
    pred_label = []
    true_label = []
    
    ## initialize counter
    total_answer_count = {"Rise":0, "Fall":0, "Steady":0}
    correct_count = {"Rise":0, "Fall":0, "Steady":0}
    accuracy_result = {}

    
    ############################################################
    def get_label(numerator, denominator, label_list):
        fraction = (numerator - denominator)  / denominator
        if fraction > 0.002:
            answer = "Rise"
            
        elif fraction < -0.002:
            answer = "Fall"
            
        else:
            answer = "Steady"
            
        label_list.append(answer)
        return answer
    ############################################################

    for i in range(1, len(y_pred_rescaled)):       
        
        pred_lbl = get_label(y_pred_rescaled[i], y_pred_rescaled[i-1], pred_label)        
        total_answer_count[pred_lbl] = total_answer_count[pred_lbl] + 1
        true_lbl = get_label(y_true_rescaled[i], y_true_rescaled[i-1], true_label)
      
        if pred_lbl == true_lbl:            
            correct_count[pred_lbl] = correct_count[pred_lbl] + 1
        del pred_lbl, true_lbl
               
    def get_accuracy(Label):
        accuracy_result[Label] = correct_count[Label] / total_answer_count[Label]
    
    for key in total_answer_count.keys():        
        get_accuracy(key)
        
    print("total accuracy : ", sum(correct_count.values()) / sum(total_answer_count.values()))      
    #OutDF = pd.DataFrame({'True' : true_label, 'Predict' : pred_label})
    print(accuracy_result)
    

    
    
    # For each output-signal.
    for signal in range(len(target_names)):
        # Get the output-signal predicted by the model.
        signal_pred = y_pred_rescaled[:, signal]
        # Get the true output-signal from the data-set.
        signal_true = y_true_rescaled[:, signal]
        # Make the plotting-canvas bigger.
        plt.figure(figsize=(15,5))
        # Plot and compare the two signals.
        plt.plot(signal_true, label='true')
        plt.plot(signal_pred, label='pred')
        # Plot grey box for warmup-period.
        #p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)
        # Plot labels etc.
        plt.ylabel(target_names[signal])
        plt.legend()
        plt.show()

if __name__=="__main__":
    np.random.seed(5)
    DB_schema = get_data()
    
    DF = pd.DataFrame(DB_schema)
    DF['Date'] = DF['Date'].apply(pd.to_datetime, errors='coerce')
    DF.index = DF['Date']
    DF = DF.sort_values(by='Date')
    DF['close'].plot()
    DF = DF[['close', 'high', 'low', 'open', 'volume']]
    
    split_idx = int(DF.shape[0]*0.6)
    Train_DF = DF[:split_idx]
    Test_DF = DF[split_idx:]
    
    Train_DF = Shift_DF(Train_DF)
    Test_DF = Shift_DF(Test_DF)
    
    x_train = Train_DF.dropna().drop('close', axis=1)
    y_train = Train_DF.dropna()[['close']]
    x_test = Test_DF.dropna().drop('close', axis=1)
    y_test = Test_DF.dropna()[['close']]
    
    x_train = x_train.values
    y_train = y_train.values
    x_test = x_test.values
    y_test = y_test.values
    
    sc = MinMaxScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    y_train = sc.fit_transform(y_train)
    y_test = sc.transform(y_test)
    
    K.clear_session()
    model = get_model(x_train)
    
    plot_comparison(model=model, start_idx=10, length=300, train=True)
    
    
    
