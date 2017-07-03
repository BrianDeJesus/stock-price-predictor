# Data science stock predictor
from matplotlib import pyplot as plt
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline as pipe
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas_datareader as dr
import argparse
import sys

def get_requested_stock_company():
    parser = argparse.ArgumentParser()

    parser.add_argument('company_abbr', type=str)
    arg = parser.parse_args()

    return arg.company_abbr.upper()


def predict_prices(dates, prices, company, x):
    print("Loading historical data graph w/ support vector machines...")
    # set up support vector models
    svr_lin = pipe(StandardScaler(), SVR(kernel='linear', C=1e3))
    svr_rbf = pipe(StandardScaler(), SVR(kernel='rbf', C=1e3, gamma=0.1))
    svr_lin.fit(dates, prices.ravel())
    svr_rbf.fit(dates, prices.ravel())
    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_lin.predict(dates), color='red', label='Linear model')
    plt.plot(dates, svr_rbf.predict(dates), color='blue', label='RBF model')
    plt.legend()
    plt.xlabel('Date interval')
    plt.ylabel('Closing Price')
    plt.title('Stock history of {} with SVR Models' .format(company))
    plt.show()
    return svr_lin.predict(x), svr_rbf.predict(x)

def analyze_stock_company(company):
    try:
        stock_data = dr.get_data_google(company)
    except dr._utils.RemoteDataError:
        print("No such company abbreviation found.")
        sys.exit()
    dates = stock_data.index.values
    prices = stock_data['Close'].values
    # reshape and change type of dates and prices arrays
    dates = np.array(dates, dtype=float)
    dates = np.reshape(dates, (len(dates),1))
    prices = np.array(prices, dtype=float)
    prices = np.reshape(prices, (len(prices),1))

    futures = []
    # getting most recent date and future dates
    futures.append(dates[len(dates)-1][0] + 0.1e+18)
    futures.append(dates[len(dates)-1][0] + 0.13e+18)
    futures.append(dates[len(dates)-1][0] + 0.15e+18)
    futures.append(dates[len(dates)-1][0] + 0.17e+18)
    futures.append(dates[len(dates)-1][0] + 0.2e+18)
    # adjusting future date data structures
    fdates = np.array(futures, dtype=float)
    fdates = np.reshape(fdates, (len(fdates),1))


    sv_lin, sv_rbf = predict_prices(dates, prices, company, fdates)
    print("Loading future outlook graph...")
    plt.plot(fdates, sv_lin, color='purple', label='Future with linear model')
    plt.plot(fdates, sv_rbf, color='green', label='Future with RBF model')
    plt.legend()
    plt.xlabel('Date interval')
    plt.ylabel('Closing Price')
    plt.title('Future outlook for {}' .format(company))
    plt.show()

def main():
    company = get_requested_stock_company()
    analyze_stock_company(company)


if __name__ == '__main__':
    main()
