#!/usr/bin/env python
# encoding: utf-8

import time
import cPickle as pkl
import pandas as pd
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer
from lasagne.nonlinearities import identity, sigmoid
from nolearn.lasagne import NeuralNet
from sklearn.datasets import make_regression
from sklearn import cross_validation

floatX = theano.config.floatX

def load_data(dataset):
    print 'Loading dataset {}'.format(dataset)
    df = pkl.load(open(dataset, 'r'))
    # target = df[['Qty Req', 'Qty Ord']]
    dfs = []
    # dfs.append(target)
    for i in range(14):
        dftmp = df[['Qty Sold', 'Qty Ord', 'Qty Rcv', 'Qty Req', 'Eff Inv']]
        dftmp.index = df.index + pd.DateOffset(days=i)
        dfs.append(dftmp)
    table = pd.concat(dfs, axis=1, join='inner')
    target = table.ix[:, :35]
    inputs = table.ix[:, 35:]
    targetorig = target.as_matrix().astype(floatX)
    inputsorig = inputs.as_matrix().astype(floatX)
    nrow, _ = inputsorig.shape
    inputs = np.zeros((nrow, 5))
    for ifeat in xrange(5):
        inputs[:, ifeat] = np.mean(inputsorig[:, range(ifeat, 35, 5)], axis=1)
    target = np.zeros((nrow, 2))
    target[:, 0] = np.mean(targetorig[:, range(1, 35, 5)], axis=1)
    target[:, 1] = np.mean(targetorig[:, range(3, 35, 5)], axis=1)
    kf = cross_validation.KFold(nrow, n_folds=10, shuffle=True)
    kfs = [(train, test) for train, test in kf]
    train, test = kfs[0]
    X_train = inputs[train].astype(floatX)
    Y_train = target[train].astype(floatX)
    X_test = inputs[test].astype(floatX)
    Y_test = target[test].astype(floatX)
    return X_train, Y_train, X_test, Y_test

def linreg(X, y):
    net = NeuralNet(
            layers=[
                ('input', InputLayer),
                ('output', DenseLayer)],
            input_shape=(None, X.shape[1]),
            output_num_units=2,
            output_nonlinearity=identity,
            regression=True,
            update_learning_rate=1e-9,
            update_momentum=0.9,
            verbose=1)
    net.fit(X, y)


def mlp(X, y):
    net = NeuralNet(
            layers=[
                ('input', InputLayer),
                ('hidden', DenseLayer),
                ('output', DenseLayer)],
            input_shape=(None, X.shape[1]),
            hidden_num_units=5,
            hidden_nonlinearity=sigmoid,
            output_num_units=2,
            output_nonlinearity=sigmoid,
            regression=True,
            update_learning_rate=1e-3,
            update_momentum=0.9,
            verbose=1)
    net.fit(X, y)


def main(model):
    X_train, Y_train, X_test, Y_test = load_data('./data.pkl')
    if model == 'linear':
        linreg(X_train, Y_train)
    elif model == 'mlp':
        mlp(X_train, Y_train)

if __name__ == '__main__':
    main('linear')
    main('mlp')

