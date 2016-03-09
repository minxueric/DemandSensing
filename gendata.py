#!/usr/bin/env python
# encoding: utf-8

import os
import cPickle as pkl
import pandas as pd

pd.options.display.max_rows = 25

def load():
    if os.path.isfile('data.pkl'):
        data = pkl.load(open('data.pkl', 'r'))
        # return data

    data = pd.read_excel('data.xls', 'Dr.RD_noName', header=0, index_col=0)
    data.fillna(value=0, inplace=True)  # fill in NA/NaN as value=0
    # print data.index        # index, i.e., DC name
    # print data.columns      # columns, i.e., variables of interest
    # print data.values       # main body of the table
    # print data.describe()   # summary of the table

    name = 'CARDINAL HEALTH-SYRACUSE'
    name = 'BORSCHOW HOSPITAL & MEDICAL SUPPLY'
    name = 'MCKESSON SPECIALTY CARE'
    # name = 'MCKESSON SPECIALTY DISTRIBUTION CORPORATION'
    data = data.loc[name, ['Day Date', 'Qty Sold', 'Qty Ord', 'Qty Rcv', 'Qty Req', 'Eff Inv']]
    # sort by Day Date
    data.sort(columns=['Day Date'], inplace=True)
    # remove duplicated datetimes
    # data.drop_duplicates(subset='Day Date', take_last=True, inplace=True)
    data = data.groupby(by='Day Date').sum()
    print data
    # fill missing datetimes
    data.reset_index(inplace=True)
    data.set_index(keys='Day Date', drop=True, inplace=True)
    idx = pd.date_range(data.index[0], data.index[-1], freq='D')
    # data = data.reindex(idx, fill_value=0)
    data = data.reindex(idx, method='ffill')

    # save and return data
    with open('data.pkl', 'wb') as f:
        pkl.dump(data, f)
    return data

def analysis(data):
    print data
    print '\nDescription of data:'
    print data.describe()
    print '\nCorrelation of data:'
    print data.corr()

if __name__ == '__main__':
    data = load()
    analysis(data)
