
import yfinance as yf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.losses import Huber


import matplotlib.pyplot as plt
import inspect



class ModelNN:
    def __init__(self,units1 = 50, dropout=0.2, units2 = 50, dense1 = 25, dense2 = 1, epochs=20, batch_size=32,test_size=0.2,sequence_length=50,optimizer="adam",):
        """Constructor method to initialize the Person object."""
        self.units1 = units1
        self.units2 = units2
        self.dropout = dropout
        self.dense1 = dense1
        self.dense2 = dense2
        self.epochs = epochs
        self.batch_size = batch_size
        self.test_size = test_size
        self.sequence_length = sequence_length
        self.optimizer = optimizer
        # self.ticker = ticker


    # def __len__(self):
    #     return len(self.items)

    def createSequence(self,data):
        x, y = [], []
        for i in range(len(data)-self.sequence_length):
            x.append(data[i:i+self.sequence_length]) #50 terms of prev day closes
            y.append(data[i+self.sequence_length])#50th term/ term after last 50 in x
        return np.array(x),np.array(y)

    def dataProcess(self):
        data = self.df[['Close']].values

        # scaler = MinMaxScaler(feature_range = (-1,1))
        # data_scaled = scaler.fit_transform(data)

        scaler = StandardScaler()  # Mean = 0, Standard Deviation = 1
        data_scaled = scaler.fit_transform(data)

        scaler = RobustScaler()  # Uses median, better for outliers
        data_scaled = scaler.fit_transform(data)


        x,y = self.createSequence(data_scaled)

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size, shuffle=False)

        self.X_train = X_train
        self.X_test = X_test
        self.scaler = scaler
        self.y_train = y_train
        self.y_test = y_test

        print('>Training and testing data acquired')


    def getData(self,ticker, start, end):
        print(f'-----------------------TICKER {ticker}')
        self.ticker = ticker
        self.start = start
        self.end = end

        # try:

        if os.path.exists(f'{ticker} - data.csv'):
            df = pd.read_csv(f'{ticker} - data.csv')

            # if df.iloc[1].str.contains("Ticker").any():
            df = df.iloc[1:]  # Drop the first row

        else:
            df = yf.download(ticker, start=start, end=end)
            df = df.reset_index()
            df.to_csv(f"{ticker} - data.csv", index=False)

        # df['Close'] = np.log1p(df['Close'])  # Apply log transformation
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')  # Convert to numeric (NaNs for errors)
        df['Close'] = np.log1p(df['Close'])  # Apply log transformation

        self.df = df
        self.dataProcess()
        return True
        # except:
        #     print("------>  there was an issue downloading and or processing the data")
        #     return False


    def defineModel(self):
        model = Sequential([
        LSTM(self.units1,return_sequences=True,input_shape=(self.sequence_length,1)),
        Dropout(self.dropout),
        LSTM(self.units2, return_sequences=False),
        Dropout(self.dropout),
        Dense(self.dense1),
        Dense(self.dense2)
        ])

        # model.compile(optimizer = 'adam', loss = "mean_squared_error")
        model.compile(loss=Huber(delta=1.5), optimizer=self.optimizer)

        model.summary()

        self.model = model

        print('>model defined successfully')

    def trainModel(self):

        weights = np.linspace(1, 2, len(self.y_train))
        weights = weights / np.mean(weights)  # Normalize so the average weight is 1

        self.history = self.model.fit(self.X_train,self.y_train, self.epochs, self.batch_size, validation_data = (self.X_test,self.y_test),sample_weight=weights)

    def testModel(self):
        predictions = self.model.predict(self.X_test)

        predictions = self.scaler.inverse_transform(predictions)

        y_test_actual = self.scaler.inverse_transform(self.y_test.reshape(-1,1))
        predicted_prices = np.expm1(predictions)  # Reverse log scaling

        self.predictions = predicted_prices
        self.y_test_actual =  np.expm1(y_test_actual)
    
    def plotResults(self):
        plt.figure(figsize=(12,6))
        plt.plot(self.y_test_actual, label="Actual Prices")
        plt.plot(self.predictions, label="Predicted Prices", linestyle='dashed')
        plt.legend()
        plt.title("Stock Price Prediction")
        plt.show()


# ticker = "AAPL"
# start_date='2010-01-01'
# end_date='2025-01-01'

# m1 = ModelNN()
# dataAQ = m1.getData(ticker,start_date,end_date)

# if dataAQ:
#     m1.defineModel()
#     m1.trainModel()
#     m1.testModel()
#     m1.plotResults()



# # print(inspect.getsource(m1))
# print(dir(m1))

# print(m1.units1)