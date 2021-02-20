#!/usr/bin/python

import math
import numpy as np
import scipy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
import re

# import autograd as ag
import torch
from torch.autograd import grad, Variable
from torch.utils.data import Sampler, Dataset, DataLoader, ConcatDataset, WeightedRandomSampler, BatchSampler
# from pyDOE2 import lhs

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import metrics
import xgboost as xgb

import timeit
import datetime
import copy

print('reimported at ', end=''); print(datetime.datetime.now())

def df_scale(data, dmean = None, dstd = None):
    
    if dmean is None:
        dmean = np.mean(data, axis=0)
    if dstd is None:
        dstd = np.std(data, axis=0)        

    return ((data-dmean) / dstd), dmean, dstd

def df_unscale(data, dmean, dstd):
    # print(data)
    # print(dmean)
    # print(dstd)
    return data*dstd + dmean

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError('Not serializable')

class dataset_from_dataframe(Dataset):
    
    def __init__(self, df, cols, transformx = None):

        if torch.cuda.is_available() == True:
            self.dtype_double = torch.cuda.FloatTensor
            self.dtype_int = torch.cuda.LongTensor
        else:
            self.dtype_double = torch.FloatTensor
            self.dtype_int = torch.LongTensor
            # self.dtype_

        self.df = df
        self.f_cols = cols
        self.t_cols = [c for c in df.columns if c not in cols]
        self.transformx = transformx        
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
                
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.df.iloc[idx]        
        features = data[self.f_cols].astype(float).values    
        label = data[self.t_cols]

        if self.transformx:
            data = self.transformx(data).copy()

        return torch.from_numpy(features).type(self.dtype_double), torch.tensor(label, dtype=torch.float32)

class FCNN(torch.nn.Module):
    
    def __init__(self, feature_dim, output_dim):
        # inherit torch.nn.Module methods
        super(FCNN, self).__init__()

        self.dtype = torch.FloatTensor
        
        d1 = 12
        d2 = 12
        d3 = 12
        d4 = 12
        # d5 = 12
        # keep this consistent
                 
        self.feature_dim = feature_dim # train_data[0][0].shape[-1]
        self.output_dim = output_dim # train_data[0][1].shape[-1]
        
        self.fc1 = torch.nn.Linear(self.feature_dim, d1)
        self.fc2 = torch.nn.Linear(d1, d2)
        self.fc3 = torch.nn.Linear(d2, d3)
        self.fc4 = torch.nn.Linear(d3, d4)
        # self.fc5 = torch.nn.Linear(d4, d5)
        self.fc6 = torch.nn.Linear(d4, self.output_dim)
       
    def init_weights(self,layer):
        # glorot initialization
        if type(layer) == torch.nn.Linear:
            torch.nn.init.xavier_normal_(layer.weight)
            layer.bias.data.fill_(0.)
        return
    
    def forward(self, f_data):

        f_data = torch.tanh(self.fc1(f_data))
        f_data = torch.tanh(self.fc2(f_data))
        f_data = torch.tanh(self.fc3(f_data))
        # f_data = torch.tanh(self.fc4(f_data))
        # f_data = torch.tanh(self.fc5(f_data))
        predictions = self.fc6(f_data)
        
        return predictions

class simpleNN(torch.nn.Module):

    def __init__(self, feature_dim, output_dim):

        super(simpleNN, self).__init__()

        self.dtype = torch.FloatTensor
        self.fcnn = FCNN(feature_dim, output_dim)
        self.fcnn.apply(self.fcnn.init_weights)

        self.optimizer = torch.optim.Adam(self.fcnn.parameters(), lr=1e-2)

        self.loss_fn = torch.nn.MSELoss()

    def train_m(self, train_loader):

        losslist = []
        labeltest = [] # measure training distribution

        for it, (data, labels) in enumerate(train_loader):

            labeltest = labeltest + [i.item() for i in labels]
            # print(data.type())

            self.optimizer.zero_grad()

            outputs = self.fcnn.forward(data)
            loss = self.loss_fn(outputs, labels)    
            loss.backward()

            self.optimizer.step()

            losslist.append(loss.detach_())

            if it % 20 == 0:
                print(losslist[-1])

            del loss

        return losslist, labeltest

    def test_m(self, test_loader):

        with torch.no_grad():
            
            losslist = []
            outputlist = []
            labellist = []

            for it, (data, labels) in enumerate(test_loader):

                # print(data.type())

                outputs = self.fcnn.forward(data)
                loss = self.loss_fn(outputs, labels)

                outputlist += outputs
                labellist += labels

                losslist.append(loss.detach_())

                if it % 20 == 0:
                    print(losslist[-1])

                del loss

        return losslist, outputlist, labellist

def run_model(train_df, test_df, cols, epochs = 1, bsize=5000):
    
    if torch.cuda.is_available() == True:
        pinning = True
    else:
        pinning = False

    # print(cols)
    train_weighted_sampler, _ = make_sampler(train_df['tm_over_tp'].values)

    train_dset = dataset_from_dataframe(train_df, cols)
    test_dset = dataset_from_dataframe(test_df, cols)

    model = simpleNN(train_dset[0][0].shape[-1], train_dset[0][1].shape[-1])
    # model.apply(model.init_weights)

    train_loader = DataLoader(train_dset, batch_size = bsize, sampler=train_weighted_sampler, pin_memory = pinning) # shuffle=True,

    train_losslist = []
    train_labels = []
    
    for epoch in range(epochs):

        start_time = timeit.default_timer()
        losslist, labeltest = model.train_m(train_loader)
        if epoch < 10:
            train_labels = train_labels + labeltest
        elapsed = timeit.default_timer() - start_time

        train_losslist += losslist

        print('Epoch %d: %f s' % (epoch, elapsed))

    plt.hist(train_labels, bins=100)
    plt.show()
    # sys.exit()

    # for weighting the test data
    # test_weighted_sampler, _ = make_sampler(test_df['tm_over_tp'].values)
    # test_loader = DataLoader(test_dset, batch_size = bsize, sampler=test_weighted_sampler, pin_memory = pinning) # shuffle=True,
    test_loader = DataLoader(test_dset, batch_size = bsize, shuffle=True, pin_memory = pinning)

    test_losslist, outputlist, labellist = model.test_m(test_loader)

    # print(train_losslist)
    # print(test_losslist)

    return model, train_losslist, test_losslist, outputlist, labellist


def regress_inputs(model, data, metadict, epochs):

    """ take in a dataframe corresponding to experiments + random samples of params, and regress using frozen model to tweak param inputs.
    """
    
    feature_cols = metadict['feature_cols']
    target_cols = metadict['target_cols']
    update_cols = metadict['update_cols']

    inputdata = data[feature_cols]
    # update_inds = [inputdata.columns.get_loc(i) for i in update_cols]
    # fixed_inds = [inputdata.columns.get_loc(i) for i in feature_cols if i not in update_cols]
    fixed_cols = [i for i in feature_cols if i not in update_cols]
    # print(update_cols)
    # print(fixed_cols)

    # print(data)
    inputdata, _, _ = df_scale(inputdata, metadict['feature_scaling_mean'], metadict['feature_scaling_std'])
    targetdata, _, _ = df_scale(data[target_cols], metadict['target_scaling_mean'], metadict['target_scaling_std'])

    # print(targetdata)
    print(inputdata[update_cols])
    inputdata[update_cols] = np.random.normal(size=(len(inputdata),len(update_cols)))
    print(inputdata[update_cols])

    optimized_inputs = inputdata.copy()

    inputfixed = torch.from_numpy(inputdata[fixed_cols].astype(float).values).type(model.dtype).requires_grad_()
    inputupdates = torch.from_numpy(inputdata[update_cols].astype(float).values).type(model.dtype).requires_grad_()
    targetdata = torch.tensor(targetdata.astype(float).values, dtype=torch.float32)

    # print('first')
    # print(inputupdates)

    optim = torch.optim.Adam([inputupdates], lr=5e-3)
    losslist = []

    for it, tt in enumerate(np.arange(epochs)):

        # print(it)
        optim.zero_grad()

        
        ### the model must have been trained with update cols on the left, fixed cols on the right
        ###
        # outputs = model.fcnn.forward(torch.cat((inputupdates, inputfixed), dim=1))
        outputs = model.fcnn.forward(torch.cat((inputupdates, inputfixed), dim=1))
        loss = model.loss_fn(outputs, targetdata)
        loss.backward()
        # loss.backward(retain_graph=True)

        optim.step()

        # print(inputupdates)
        # print(torch.mean(inputupdates,0))
        # print(torch.ones(inputupdates.size()))
        # with torch.no_grad():
        # print(torch.mean(inputupdates,0).detach_())
        inputupdates = torch.mean(inputupdates,0).detach_() * torch.ones(inputupdates.size())
        inputupdates.requires_grad_()
        optim = torch.optim.Adam([inputupdates], lr=5e-3)
        # print(inputupdates)
        # print(torch.mean(inputupdates,0))
        

        # print(inputupdates)
        losslist.append(loss.detach_())

        # sys.exit()
    outputs = model.fcnn.forward(torch.cat((inputupdates, inputfixed), dim=1))
    
    inputdata = torch.cat((inputupdates, inputfixed), dim=1).detach().numpy()
    inputdata = df_unscale(inputdata, metadict['feature_scaling_mean'], metadict['feature_scaling_std'])
    outputs = df_unscale(outputs.detach().numpy(), metadict['target_scaling_mean'], metadict['target_scaling_std'])
    
    optimized_inputs = pd.DataFrame(inputdata, columns=optimized_inputs.columns)
    # print(optimized_inputs)
    # print(outputs)
    optimized_inputs[target_cols[0]] = outputs

    # print(optimized_inputs)

    # sys.exit()

    return optimized_inputs, losslist


def random_forest(train_df, test_df, metadict):

    # rgr = RandomForestRegressor(n_estimators=200, min_samples_split=6, max_depth=8)
    # rgr.fit(train_df[metadict['feature_cols']], train_df[metadict['target_cols']].values.ravel())
    # ytrains = rgr.predict(train_df[metadict['feature_cols']])
    # y_pred = rgr.predict(test_df[metadict['feature_cols']])
    # train_mse = metrics.mean_squared_error(train_df[metadict['target_cols']].values.ravel(), ytrains)
    # mse = metrics.mean_squared_error(test_df[metadict['target_cols']].values.ravel(), y_pred)
    # print("Train MSE:", train_mse)
    # print("Test MSE:", mse)

    # gbr = GradientBoostingRegressor(n_estimators=100, subsample=0.8, min_samples_split=10, max_depth=10)
    # gbr.fit(train_df[metadict['feature_cols']], train_df[metadict['target_cols']].values.ravel())
    # gbytrains = gbr.predict(train_df[metadict['feature_cols']])
    # gby_pred = gbr.predict(test_df[metadict['feature_cols']])
    # gbtrain_mse = metrics.mean_squared_error(train_df[metadict['target_cols']].values.ravel(), gbytrains)
    # gbmse = metrics.mean_squared_error(test_df[metadict['target_cols']].values.ravel(), gby_pred)
    # print("gb Train MSE:", gbtrain_mse)
    # print("gb Test MSE:", gbmse)
    # pred = gby_pred
    # test_labels = test_df[metadict['target_cols']].values.ravel()
    # mse = gbmse


    xgbr = xgb.XGBRegressor(
        gamma=1,                 
        learning_rate=0.005,
        max_depth=10,
        n_estimators=160,                                                                    
        subsample=1.,
        random_state=34
        ) 

    xgbr.fit(train_df[metadict['feature_cols']], train_df[metadict['target_cols']])
    xgbrytrains = xgbr.predict(train_df[metadict['feature_cols']])
    xgbr_preds = xgbr.predict(test_df[metadict['feature_cols']])
    xgbrtrain_mse = metrics.mean_squared_error(train_df[metadict['target_cols']].values.ravel(), xgbrytrains)
    xgbrmse = metrics.mean_squared_error(test_df[metadict['target_cols']].values.ravel(), xgbr_preds)
    print("xgb Train MSE:", xgbrtrain_mse)
    print("xgb Test MSE:", xgbrmse)
    pred = xgbr_preds
    test_labels = test_df[metadict['target_cols']].values.ravel()
    mse = xgbrmse

    # print(train_df[metadict['target_cols']])
    # print(train_df[metadict['feature_cols']])
    # print(train_df[metadict['target_cols']].values)

    

    return pred, test_labels, mse

def make_sampler(data):

    counts, bins = np.histogram(data, bins=200)
    # print(counts)
    # print(bins)

    # print(np.digitize(data, bins))
    # print(np.amax(np.digitize(data, bins[1:], right=True)))
    # print(np.amin(np.digitize(data, bins[1:], right=True)))
    weights = np.zeros(len(counts))
    weights[counts!=0] = 1. / counts[counts != 0]
    # print(np.sum(weights))
    # print(weights[counts==0])
    # sample_weights
    sample_weights = weights[np.digitize(data,bins[1:], right=True)]

    # print(np.sum(sample_weights==np.amin(weights)))

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    # sys.exit()

    return sampler, sample_weights #counts / np.sum(counts)


if __name__ == '__main__':

    cwd = os.getcwd()
    pd.set_option('display.expand_frame_repr', False, 'display.max_columns', None)

    # df = pd.read_csv('./stiff_results/constant/static_LHS_SG_SR_ng.csv')
    # df = pd.read_csv('./stiff_results/staticTS/static_LHS_SG_SR_ng_newdyn.csv')
    df = pd.read_csv('./stiff_results/dynamicTS/energy_dependent_LHS_ng_newdyn.csv')

    # print(df.loc[df['ptime'] != df['prime_time']])
    # (df['ptime'] - df['prime_time']).hist(bins=40)
    # plt.show()

    # feature_cols = ['tau', 'tau_F', 'tau_SG', 'tau_SR', 'm0', 'n', 'ptime', 'mem_stiff', 'prime_stiff']
    # feature_cols = ['tau', 'tau_F', 'tau_R0', 'TV0SG', 'TV0SR', 'm0','n', 'ptime', 'mem_stiff', 'prime_stiff']
    # static
    update_features = ['tau', 'tau_F', 'tau_SG', 'tau_SR', 'm0', 'n']
    # energy dep
    update_features = ['tau', 'tau_F', 'tau_R0', 'TV0SG', 'TV0SR', 'm0','n']
    fix_features = ['ptime', 'mem_stiff', 'prime_stiff']
    feature_cols = update_features + fix_features

    df['tm_over_tp'] = df['mem_time'] / df['prime_time']

    # print(df.loc[df['mem_time'].isna()])
    target_cols = ['tm_over_tp']

    df = df[feature_cols + target_cols]
    df, d_mean, d_std = df_scale(df)

    # df['tm_over_tp'].hist(bins=200)
    # plt.show()
    # sys.exit()

    data_fraction = 1. # to speed up testing, use some random fraction of images.
    train_fraction = 0.8
    test_fraction = 1 - train_fraction
    bsize = 200
    epochs = 10

    ids = df.index.values
    np.random.shuffle(ids)

    ushuff = ids[:int(len(ids)*data_fraction)]
    utrain = ushuff[:int(len(ushuff) * train_fraction)]
    utest = ushuff[int(len(ushuff) * train_fraction):]

    train_df = df.iloc[utrain].reset_index(drop=True)
    test_df = df.iloc[utest].reset_index(drop=True)

    print(train_df.head(5))  

    model_metadict = {
                    'feature_cols': feature_cols,
                    'target_cols': target_cols,
                    'update_cols': update_features,
                    'feature_scaling_mean': d_mean[feature_cols].values,
                    'feature_scaling_std': d_std[feature_cols].values,
                    'target_scaling_mean': d_mean[target_cols].values,
                    'target_scaling_std': d_std[target_cols].values,
                    }

    _, sweights = make_sampler(train_df[target_cols].values)
    # print(np.shape(sweights))
    # print(torch.as_tensor(sweights.flatten(), dtype=torch.double))
    # sys.exit()
    balanced_inds = torch.multinomial(torch.as_tensor(sweights.flatten(), dtype=torch.double), len(sweights), True)
    print(balanced_inds.numpy())
    # print(np.sum(sweights.flatten()))
    # balanced_inds = np.random.choice(np.arange(len(train_df[target_cols].values)), len(train_df[target_cols].values), True, sweights.flatten())
    weighted_df = train_df.iloc[balanced_inds]
    # weighted_df[target_cols].hist()
    # plt.show()
    # sys.exit()

    pred, test_labels, mse = random_forest(weighted_df, test_df, model_metadict)
    fig3, ax3 = plt.subplots(1,1, figsize=(6,6))

    ax3.scatter(pred, test_labels)
    ax3.set_xlabel('predicted')
    ax3.set_ylabel('true')

    ax3.plot([np.amin(test_labels), np.amax(test_labels)], [np.amin(test_labels), np.amax(test_labels)], color='r')
    ax3.set_xlim([np.amin(test_labels), np.amax(test_labels)])
    ax3.set_ylim([np.amin(test_labels), np.amax(test_labels)])

    plt.show()

    sys.exit()
    
    # scale the domain and not the training data.. then scale the equation when calculating loss
    model, train_loss, test_loss, outputlist, labellist = run_model(train_df, test_df, feature_cols, epochs, bsize)

    outputs = np.array([i.item() for i in outputlist])
    labels = np.array([i.item() for i in labellist])


    # model_metadict = {
    #                 'feature_cols': feature_cols,
    #                 'target_cols': target_cols,
    #                 'update_cols': update_features,
    #                 'feature_scaling_mean': d_mean[feature_cols].values,
    #                 'feature_scaling_std': d_std[feature_cols].values,
    #                 'target_scaling_mean': d_mean[target_cols].values,
    #                 'target_scaling_std': d_std[target_cols].values,
    #                 }
    # # print(type(outputs))
    # # print(outputs)
    # # print(d_mean[target_cols].values)
    # # print(d_std[target_cols].values)
    outputs = df_unscale(outputs, d_mean[target_cols].values, d_std[target_cols].values)
    labels = df_unscale(labels, d_mean[target_cols].values, d_std[target_cols].values)

    train_loss = [i.item() for i in train_loss]
    test_loss = [i.item() for i in test_loss]

    fig, ax = plt.subplots(1,2, figsize=(12,6))

    avg_train_loss = moving_average(train_loss, int(len(train_df)/bsize))
    avg_test_loss = moving_average(test_loss, int(len(test_df)/bsize))

    ax[0].plot(np.arange(len(train_loss)), train_loss)
    ax[0].plot(np.arange(len(test_loss)), test_loss)

    ax[1].scatter(outputs, labels)
    ax[1].set_xlabel('predicted')
    ax[1].set_ylabel('true')

    ax[1].plot([np.amin(labels), np.amax(labels)], [np.amin(labels), np.amax(labels)], color='r')
    ax[1].set_xlim([np.amin(labels), np.amax(labels)])
    ax[1].set_ylim([np.amin(labels), np.amax(labels)])

    torch.save(model.state_dict(), './models/FCNN_trained.out')

    plt.tight_layout()
    # plt.show()        
    
    data = pd.read_csv('./experiments/energy_dep_experimental_DF.csv')

    # print(data)
    epochs = 1000

    data['tm_over_tp'] = data['mem_time'] / data['prime_time']

    model_metadict['update_cols'] = update_features # 'prime_time'
    optimized_inputs, losslist = regress_inputs(model, data, model_metadict, epochs)

    fig2, ax2 = plt.subplots(1,2, figsize=(8,6))

    ax2[0].plot(np.arange(len(losslist)), losslist)

    print(optimized_inputs)

    print('hi')
    model2 = simpleNN(len(feature_cols), len(target_cols))
    model2.load_state_dict(copy.deepcopy(torch.load('./models/FCNN_trained.out')))
    model2.eval()

    optimized_inputs, losslist = regress_inputs(model2, data, model_metadict, epochs)

    plt.show()