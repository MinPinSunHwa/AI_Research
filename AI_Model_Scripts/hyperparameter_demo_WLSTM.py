# accuracy for different models:
# WLSTM: 0.71(loss=0.4)
# RNN: 0.66        
# LSTM: 0.66 (loss=0.6)
# GRU: 0.6 (loss=0.57)      w-GRU(Wavelet Transform with GRU): 0.735 
# twoRNN: 0.63 (loss=0.55)      W-twoRNN: 0.73        
# twoLSTM: 0.5 (loss=0.9, need more steps to train)    w-twoLSTM: 0.71
# twoGRU: 0.63 (loss=0.5)   w-twoGRU: 0.74
# ThreeLSTM: 0.39 (loss=1.08)  w-ThreeLSTM: 0.73
# Conclusion: The model with data denoise-processed exhibit

import tensorflow as tf
import json
import pandas as pd
import matplotlib.pyplot as plt
import math, pytz
import sklearn
from sklearn.ensemble import RandomForestRegressor
from pywt import wavedec, threshold, waverec
from pytz import all_timezones, timezone, utc
from statsmodels.robust import mad
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta



############################# Model 1 ####################################
class RNN_Model_build(object):
    def __init__(self):
        tf.reset_default_graph()  # Reset existing graph
        DF = Get_Mean_Median()
        self.dm = Data_manager(DF)
        self.dm.data_preprocess()

    def show_as_Seoul_time(self, UTC):
        Seoul = timezone('Asia/Seoul')
        UTCTime = utc.localize(UTC)
        SeoulTime = UTCTime.astimezone(Seoul)
        return self.dm.Datetime_to_Str(SeoulTime)

    def RNN_cell(self):
        cell = tf.contrib.rnn.BasicRNNCell(num_units=self.dm.hidden_dim)
        return cell

    def Architecture(self, modelDescript, modelName="Model_02"):
        sess = tf.Session()


        ### Model ###
        num_step = 30000
        # learning rate decay
        nbatch,_, _ = self.dm.x_train.shape
        learning_rate = tf.train.exponential_decay(0.13, num_step, nbatch, 0.8, staircase=True)

        X = tf.placeholder(tf.float32, [self.dm.batch_size, self.dm.window_length, self.dm.data_dim])
        Y = tf.placeholder(tf.float32, [self.dm.batch_size, self.dm.output_dim])

        Cell = self.RNN_cell()
        outputs, _states = tf.nn.dynamic_rnn(Cell, X, dtype=tf.float32)

        Y_pred = tf.contrib.layers.fully_connected(outputs[:,-1], self.dm.output_dim, activation_fn=None)  # We only interest the last output value

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_pred,labels=Y))

        # To fix the over fitting problem, introduce the L2 regularization to the loss functio
        #regularization_rate=0.1
        #regularizer=tf.contrib.layers.l2_regularizer(regularization_rate)
        #regulation=tf.contrib.layers.apply_regularization(regularizer,weight_list=None)
        #regularization=regularizer(weights1)+regularizer(weighits2)   ### How to find?
        #loss=loss+regularization

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_constant = 0.6  # Choose an appropriate one.
        loss = loss + reg_constant * sum(reg_losses)



        # calculate accuracy
        correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))   # bool vector consisting of true of false
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))        # changing the bool into real number, then calculating the mean value, which this the accuracy of this batch.

        optimizer = tf.train.AdamOptimizer(learning_rate)
        train = optimizer.minimize(loss)
        tf.glorot_uniform_initializer()
        print("%s graph complete" % (modelName))



        ### Session ###
        sess = tf.Session()
         # tf.glorot_uniform_initializer()   tf.global_variables_initializer()
        tf.set_random_seed(2)
        sess.run(tf.global_variables_initializer())

        validate_feed={X: self.dm.x_val, Y: self.dm.y_val}
        accuracy_train=[]
        accuracy_val=[]
        loss_shown=[]
        step=[]
        for i in range(num_step):
            _, l, accuracy_print = sess.run([train, loss,accuracy], feed_dict={X: self.dm.x_train, Y: self.dm.y_train})
            validate_acc=sess.run(accuracy,feed_dict=validate_feed)
            if (i % nbatch == 0):
                step.append(i)
                accuracy_train.append(accuracy_print)
                accuracy_val.append(validate_acc)
                loss_shown.append(l)
                print("Step %d  :  Loss %.9f, accuracy for recent batch % 9f" % (i, l,accuracy_print))
                print("After %d training step(s), validation accuracy" "using average model is %g" %(i,validate_acc))

        print("Training Done!")
        y_pred = sess.run(Y_pred, feed_dict={X: self.dm.x_val})
        y_pred = tf.nn.softmax(y_pred)
        plt.figure()
        plot1 = plt.plot(step, accuracy_train, label='train data accuracy',color='green')
        plot2 = plt.plot(step, accuracy_val, label='validation data accuracy',color='red')
        plot3=plt.plot(step,loss_shown,label='loss',color='black')
        plt.title("Accuracy")
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.ylim(0.0, 1.0)
        plt.legend()
        plt.show()


############################# Model 2 ####################################
class LSTM_Model_build(RNN_Model_build, object):

    def RNN_cell(self):
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.dm.hidden_dim)
        return cell


############################# Model 3 ####################################
class GRU_Model_build(RNN_Model_build, object):

    def RNN_cell(self):
        cell = tf.contrib.rnn.GRUCell(num_units=self.dm.hidden_dim)
        return cell


############################# Model 4 ####################################
class Two_Layer_RNN(RNN_Model_build, object):

    def Single_cell(self):
        cell = tf.contrib.rnn.BasicRNNCell(num_units=self.dm.hidden_dim)
        return cell

    def RNN_cell(self):
        stacked_cell = tf.contrib.rnn.MultiRNNCell([self.Single_cell() for _ in range(2)], state_is_tuple=True)
        return stacked_cell


############################# Model 5 ####################################
class Two_Layer_LSTM(Two_Layer_RNN, object):

    def Single_cell(self):
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.dm.hidden_dim)
        dropout_lstm=tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=0.3)
        stacked_lstm=tf.contrib.rnn.MultiRNNCell([dropout_lstm], state_is_tuple=True)

        return stacked_lstm


############################# Model 6 ####################################
class Two_Layer_GRU(Two_Layer_RNN, object):

    def Single_cell(self):
        cell = tf.contrib.rnn.GRUCell(num_units=self.dm.hidden_dim)
        return cell


############################# Model 7 ####################################
class Three_Layer_LSTM(Two_Layer_LSTM, object):

    def RNN_cell(self):
        stacked_cell = tf.contrib.rnn.MultiRNNCell([self.Single_cell() for _ in range(3)], state_is_tuple=True)
        return stacked_cell

############################# Model 11 ####################################
class Four_Layer_LSTM(Three_Layer_LSTM, object):

    def RNN_cell(self):
        stacked_cell = tf.contrib.rnn.MultiRNNCell([self.Single_cell() for _ in range(4)], state_is_tuple=True)
        return stacked_cell


############################# Model 8 ####################################
class Wavelet_LSTM(LSTM_Model_build, object):

    def __init__(self):
        DF = Get_Mean_Median()
        self.dm = Data_manager(DF)
        self.dm.x = self.Wavelet_Transform(self.dm.x)
        self.dm.y = self.Wavelet_Transform(self.dm.y)
        self.dm.data_preprocess()

    def Wavelet_Transform(self, DF):
        for i in DF.columns.values.tolist():
            print('Wavelet Transform factor:', i)

            x_data = DF[i]
            x_data = x_data.values

            # decomposition
            coeff = wavedec(x_data, 'haar', level=2)

            # wavelet denoising by means of thresholding
            sigma = mad(coeff[-1])
            uthresh = sigma * np.sqrt(2 * np.log(len(x_data)))  # calculates thresholding in the Wavelet domain

            coeff[1:] = (threshold(i, value=uthresh, mode="soft") for i in coeff[1:])  # Shrink coefficients by thresholding
            # Reconstruct the signal
            x_ = waverec(coeff, 'haar')
            try:
                DF[i] = x_
            except ValueError as ve:
                DF[i] = x_[:-1]

        return DF


############################# Model 9 ####################################
class Differentiate_LSTM(LSTM_Model_build, object):

    def __init__(self):
        DF = Get_Mean_Median()
        DF = DF.diff()[1:]
        self.dm = Data_manager(DF)
        self.dm.data_preprocess()

    def get_label(self, Grad_pred, Grad_true):

        if Grad_pred > 30:
            answer = "Rise"

        elif Grad_pred < -30:
            answer = "Fall"

        else:
            answer = "Steady"

        return answer


############################# Model 10 ###################################
class Random_Forest_Regressor(object):

    def __init__(self):
        DF = Get_Mean_Median()
        self.dm = Data_manager(DF)
        self.dm.data_preprocess()

    def show_as_Seoul_time(self, UTC):
        Seoul = timezone('Asia/Seoul')
        UTCTime = utc.localize(UTC)
        SeoulTime = UTCTime.astimezone(Seoul)
        return self.dm.Datetime_to_Str(SeoulTime)

    def Architecture(self, modelDescript, modelName="Model_10"):
        nbatchx, nwx, ndx = self.dm.x_train.shape

        x_train = self.dm.x_train.reshape((nbatchx, nwx * ndx))
        y_train = self.dm.y_train
        ### Model ###
        model = RandomForestRegressor(n_estimators=100, max_depth=20)
        model.fit(x_train, y_train)

        nbatch_, nw_, nd_ = self.dm.x_val.shape
        x_val = self.dm.x_val.reshape((nbatch_, nw_ * nd_))
        model_pred = model.predict(x_val)  # .reshape(-1,1)
        # X_Ticks = Test_DF.dropna().index
        y_pred_rescaled = self.dm.scalerY.inverse_transform(model_pred.reshape(-1, 1))
        y_true_rescaled = self.dm.scalerY.inverse_transform(self.dm.y_val.reshape(-1, 1))

        pred_lbl = self.dm.get_label(y_pred_rescaled[-1], y_true_rescaled[-1])
        true_lbl = self.dm.get_label(y_true_rescaled[-1], y_true_rescaled[-1])

        # print("Model forecast price at %s as %.3f. Real price is %.3f" %(self.dm.PredictTime, y_pred_rescaled[-1], y_true_rescaled[-1]) )
        Earlist = self.dm.Datetime_to_Str(self.dm.latest_x.index[0])
        Latest = self.dm.Datetime_to_Str(self.dm.latest_x.index[-1])  # The last time of input data
        PredictTime = self.dm.y.index[-1] + timedelta(seconds=7200)
        PredictTime_in_Korea = self.show_as_Seoul_time(PredictTime)
        PredictTime = self.dm.Datetime_to_Str(PredictTime)  # 2 hrs later from the latest time

        nbatch_t, nw_t, nd_t = self.dm.x_test.shape
        x_test = self.dm.x_test.reshape((nbatch_t, nw_t * nd_t))
        y_forecast = model.predict(x_test)
        y_forecast = self.dm.scalerY.inverse_transform(y_forecast.reshape(-1, 1))
        current_Price = self.dm.latest_x['close'][-1]
        forecast_Price = y_forecast[-1][0]
        forecast_lbl = self.dm.get_label(y_forecast[-1], current_Price)
        # print("Price : %.3f ---> %.3f [%s] expected" %(current_Price, forecast_Price, forecast_lbl))
        outDict = {"model_name": modelName, "Model_description": modelDescript, "pred_time": PredictTime,
                   "pred_time(Seoul)": PredictTime_in_Korea, "pred_price": str(forecast_Price),
                   "pred_movement": forecast_lbl, "current_time": Latest, "current_price": current_Price}
        with open('./Json_Predict/%s.json' % (modelName), 'w') as of:
            json.dump(outDict, of)

        self.plot_comparison(-19, -1, y_pred_rescaled, y_true_rescaled, self.dm.latest_x.index, modelName)


import numpy as np
import pandas as pd
import quandl
from sklearn.preprocessing import MinMaxScaler



class Data_manager(object):

    def __init__(self, DF):
        np.random.seed(1)
        DF = DF.dropna()
        DF = DF.drop_duplicates()
        self.x = DF
        self.y = DF
        self.scalerX = MinMaxScaler(feature_range=(-1, 1))
        self.scalerY = MinMaxScaler(feature_range=(-1, 1))
        self.data_dim = self.x.shape[-1]
        self.output_dim = 3
        self.batch_size = None
        self.window_length = 5
        self.percent_train = 0.6
        self.hidden_dim = 5
        # extracting a series to test the accuracy (For the real time case, change this one to the latest time series. )
        self.latest_x = DF[-200:-self.window_length]



    def Datetime_to_Str(self, DATE):
        return DATE.strftime("%Y-%m-%d %H:%M:%S")

    def get_label(self, numerator, denominator):
        fraction = (numerator - denominator) / denominator
        if fraction > 0.003:
            answer = "Rise"

        elif fraction < -0.003:
            answer = "Fall"

        else:
            answer = "Steady"

        return answer

    def data_preprocess(self):
        timesteps = 4
        print("----------------------- Data preprocess -------------------------")
        self.x = self.x.ix[:-timesteps, :]
        movement = self.y['close']
        length_change=len(movement.tolist())
        movement_collection=[]

       # output movement
        for i in list(range(length_change-4)):
            movement[i]=(movement[i+4]/movement[i])-1
            if movement[i] > 0.003:
                movement_collection.append ([1,0,0])
            elif movement[i] < -0.003:
                movement_collection.append([0, 0, 1])
            else:
                movement_collection.append([0, 1, 0])
        movement_collection=pd.DataFrame(movement_collection)


        self.y = self.y.ix[:-timesteps, :]
        xx = self.scalerX.fit_transform(self.x.values)
        yy=movement_collection.values
        final_x = self.scalerX.fit_transform(self.latest_x.values)
        arrayListX = []
        arrayListY = []
        for i in range(0, len(self.x) - self.window_length, self.window_length):
            _x = xx[i:i + self.window_length]
            _y = yy[i + self.window_length]  # Next close price
            arrayListX.append(_x)
            arrayListY.append(_y)

        # train/test split
        train_size = int(len(arrayListX) * self.percent_train)
        test_size = len(arrayListX) - train_size
        self.x_train, self.x_val = np.array(arrayListX[0:train_size]), np.array(arrayListX[train_size:len(self.x)])
        self.y_train, self.y_val = np.array(arrayListY[0:train_size]), np.array(arrayListY[train_size:len(self.y)])
        self.x_test = np.array([final_x])
        print("X train dimension : ", self.x_train.shape)
        print("X validation dimension : ", self.x_val.shape)
        print("y train dimension : ", self.y_train.shape)
        print("y validation dimension : ", self.y_val.shape)
        print("X test dimension : ", self.x_test.shape)
        Earlist = self.Datetime_to_Str(self.x.index[-self.window_length])
        Latest = self.Datetime_to_Str(self.x.index[-1])  # The last time of input data
        self.PredictTime = self.Datetime_to_Str(self.y.index[-1])  # 2 hrs later from the latest time
        # print("To forecast price at %s, data from %s to %s will be validated" %(self.PredictTime, Earlist, Latest))





import numpy as np
import pandas as pd
from pymongo import *
from datetime import time, tzinfo, timedelta, datetime

client = MongoClient('52.79.239.183', 27017)
print("DB connection complete!!")


def Get_each_Coin_DF(coinName):  # 得到输入数据
    DB_Coin = client[coinName]
    Collection = DB_Coin['BTC/USD_30MIN']
    DB_schema = []
    for collect in Collection.find():
        temp_record = {}
        temp_record["%s_open" % coinName] = collect['price_open']
        temp_record["%s_high" % coinName] = collect['price_high']
        temp_record["%s_low" % coinName] = collect['price_low']
        temp_record["%s_close" % coinName] = collect['price_close']

        temp_record['Date'] = collect['time_period_end']
        temp_record['label'] = coinName
        temp_record["%s_vol" % coinName] = collect['volume_traded']
        DB_schema.append(temp_record)
        del temp_record

    DF = pd.DataFrame(DB_schema)
    DF['Date'] = DF['Date'].apply(pd.to_datetime,
                                  errors='coerce')  ###   pd.to_datetime: python的datetime日期时间就格式化striptime
    DF.index = DF['Date']
    DF = DF.sort_values(by='Date')  ###   排序，根据日期
    DF = DF[["%s_open" % coinName, "%s_high" % coinName, "%s_low" % coinName, "%s_close" % coinName,
             "%s_vol" % coinName]]  # remove Date Column
    return DF


def Get_Mean_Median():
    BinanceDF = Get_each_Coin_DF("BINANCE")
    HuobiDF = Get_each_Coin_DF("HUOBI")
    OkexDF = Get_each_Coin_DF("OKEX")
    HitbtcDF = Get_each_Coin_DF("HITBTC")
    BitfinexDF = Get_each_Coin_DF("BITFINEX")

    JoinedDF = pd.merge(BinanceDF, HuobiDF, how='inner', left_index=True, right_index=True)
    for df in [OkexDF, HitbtcDF, BitfinexDF]:
        JoinedDF = pd.merge(JoinedDF, df, how='inner', left_index=True, right_index=True)

    Top5 = ['BINANCE', 'HUOBI', 'OKEX', 'HITBTC', 'BITFINEX']
    VolDF = JoinedDF[[Name + "_vol" for Name in Top5]]
    CloseDF = JoinedDF[[Name + "_close" for Name in Top5]]
    OpenDF = JoinedDF[[Name + "_open" for Name in Top5]]
    LowDF = JoinedDF[[Name + "_low" for Name in Top5]]
    HighDF = JoinedDF[[Name + "_high" for Name in Top5]]
    MeanMedian = [OpenDF.mean(1), HighDF.mean(1), LowDF.mean(1), CloseDF.mean(1), VolDF.median(1)]
    DF = pd.concat(MeanMedian, axis=1)
    DF.columns = ['open', 'high', 'low', 'close', 'volume']

    return DF


Get_Mean_Median()



from time import time, monotonic
import tensorflow as tf




def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Elapsed Time :")
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


# In[4]:


start = monotonic()



WLSTM = Wavelet_LSTM()
WLSTM.Architecture("Wavelet Transform-LSTM", "Model_08")
RNN = RNN_Model_build()
RNN.Architecture("Simple RNN model",  "Model_01")
LSTM = LSTM_Model_build()
LSTM.Architecture("Simple LSTM model", "Model_02")
GRU = GRU_Model_build()
GRU.Architecture("Simple GRU model", "Model_03")
TwoRNN = Two_Layer_RNN()
TwoRNN.Architecture("2-layer-RNN", "Model_04")
TwoLSTM = Two_Layer_LSTM()
TwoLSTM.Architecture("2-layer-LSTM", "Model_05")
TwoGRU = Two_Layer_GRU()
TwoGRU.Architecture("2-layer-GRU", "Model_06")
ThreeLSTM = Three_Layer_LSTM()
ThreeLSTM.Architecture("3-layer-LSTM", "Model_07")


end = monotonic()
timer(start, end)
print(start,end)


end = monotonic()
timer(start, end)
print(start,end)
