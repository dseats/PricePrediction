This is a simple LSTM layer NN used to predict stock prices based on a number of previous closing data.

There is a function to optimize/tune the hyper parameters which is currently not called.

Currently using a Robust scaler to handle outliers and price jumps.

Using Hubber loss to account for volatility and outliers as well. 

Most recent data is weighted more. 

Training data is closing stock data that is acquired through yahoo finance API. The current ticker, start and end dates can be swapped for another ticker and date.
Once the data has been collected it is written to a csv for fast loading on future model iterations.
