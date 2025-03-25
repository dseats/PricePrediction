
import yfinance as yf
import pandas as pd
from sklearn.model_selection import ParameterGrid
import modelDef
import matplotlib.pyplot as plt



def optimizeModel(params_orig,ticker,start_date,end_date):
    param_grid = {
    'sequence_length': [30, 50, 75, 100],
    'dropout': [0.2, 0.3, 0.4],
    'optimizer': ['adam', 'rmsprop'],
    'units': [50, 100]
    }
    for params in ParameterGrid(param_grid):

        print(f"Model has current params --> {params}\n")
        params_orig['sequence_length'] = params['sequence_length']
        params_orig['dropout'] = params['dropout']
        params_orig['optimizer'] = params['optimizer']
        params_orig['units1'] = params['units']
        params_orig['units1'] = params['units']

        m1 = modelDef.ModelNN(**params_orig)
        dataAQ = m1.getData(ticker,start_date,end_date)
        m1.defineModel()
        m1.trainModel()

        loss = m1.model.evaluate( m1.X_test,  m1.y_test, verbose=0)
        if loss < best_score:
            best_score = loss
            best_model = m1
            best_params = params

    print(f'Best parameters -  {best_params}\n')
    print(f'Best loss score -  {best_score}\n')


def makeRunModel(params_orig,ticker,start_date,end_date):

    m1 = modelDef.ModelNN(**params_orig)
    dataAQ = m1.getData(ticker,start_date,end_date)

    if dataAQ:
        m1.defineModel()
        m1.trainModel()
        m1.testModel()
        m1.plotResults()

        priceDiff = abs(m1.predictions - m1.y_test_actual)

        plt.plot(priceDiff)
        plt.show()


if __name__ == "__main__":
    ticker = "AAPL"
    start_date='2010-01-01'
    end_date='2025-01-01'
    params_orig = {
        'units1' : 50, 
        'dropout': 0.2, 
        'units2' : 50, 
        'dense1' : 25, 
        'dense2' : 1, 
        'epochs': 20, 
        'batch_size' : 32,
        'sequence_length': 50,
        'test_size': 0.2,
        'optimizer':"adam"
    }

    makeRunModel(params_orig,ticker,start_date,end_date)