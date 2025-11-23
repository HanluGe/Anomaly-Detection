# -*- coding: utf-8 -*-
"""
Created on Wed October 8 09:42:19 2025

@author: hongs; adapted from the orignal copy by Mr. GUAN Chenjiong, 2025.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random

import seaborn as sns
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.lay1 = nn.GRU(20, 40, num_layers=1, batch_first=True)
        self.lay2 = nn.Sequential(nn.Linear(40, 40), nn.LeakyReLU(0.01), nn.Linear(40, 40))
        self.lay3 = nn.GRU(40, 40, num_layers=1, batch_first=True)
        self.lay4 = nn.Sequential(nn.Linear(40, 40), nn.LeakyReLU(0.01), nn.Linear(40, 40))
        self.lay5 = nn.GRU(40, 19, num_layers=1, batch_first=True) # note: one layer should have less than 20 nodes to avoid data repeatation
        self.lay6 = nn.Sequential(nn.Linear(19, 40), nn.LeakyReLU(0.01), nn.Linear(40, 40))
        self.lay7 = nn.GRU(40, 20, num_layers=1, batch_first=True)
        self.lay8 = nn.Sequential(nn.Linear(40, 40), nn.LeakyReLU(0.01), nn.Linear(40, 20))
    
    #forward propagation
    def forward(self, x):
        y, _ = self.lay1(x)
        z = self.lay2(y)
        u, _ = self.lay3(z)
        v = self.lay4(u)
        w, _ = self.lay5(v)
        o = self.lay6(w)
        p, _ = self.lay7(o)
        #q = self.lay8(p)
        return p

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.lay1 = nn.GRU(20, 40, num_layers=2, batch_first=True)
        self.lay2 = nn.Sequential(nn.Linear(40, 40), nn.LeakyReLU(0.01), nn.Linear(40, 40))
        self.lay3 = nn.GRU(40, 40, num_layers=1, batch_first=True)
        self.lay4 = nn.Sequential(nn.Linear(40, 40), nn.LeakyReLU(0.01), nn.Linear(40, 40))
        self.lay5 = nn.GRU(40, 40, num_layers=1, batch_first=True)
        self.lay6 = nn.Sequential(nn.Linear(40, 40), nn.LeakyReLU(0.01), nn.Linear(40, 40))
        self.lay7 = nn.GRU(40, 40, num_layers=1, batch_first=True)
        self.lay8 = nn.Sequential(nn.Linear(40, 40), nn.LeakyReLU(0.01), nn.Linear(40, 1))
        self.drop = nn.Dropout(0.15)

    #forward propagation
    def forward(self, x):
        y, _ = self.lay1(x)
        z = self.lay2(y)
        v, _ = self.lay3(z)
        u = self.lay4(v)
        w, _ = self.lay5(u)
        r = self.lay6(w)
        s, _ = self.lay7(r)
        t = self.lay8(s)
        return torch.sigmoid(t[:, -1])

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def prepareMinutelyData(df:pd.DataFrame, tradingDays:list):
    if df.empty:
        return None
    else:
        ###Some clean up
        df['bfValue'] = df['lastPx']*df['size']
        df['bfValue'] = df['bfValue'].ffill()
        df['cumValue'] = df.groupby('date')['bfValue'].cumsum()
        df=df[df['SP1']>0]
        df=df[df['BP1']>0]
        df=df[df['SP1']-df['BP1']>0]
        for i in range(1, 6):
            df['SP{}'.format(str(i))] = df['SP{}'.format(str(i))]/100
            df['BP{}'.format(str(i))] = df['BP{}'.format(str(i))]/100
            df['SV{}'.format(str(i))] = df['SV{}'.format(str(i))]*1000
            df['BV{}'.format(str(i))] = df['BV{}'.format(str(i))]*1000
        df['lastPx'] = df['lastPx']/100
        df['size'] = df['size']*1000
        df['volume'] = df['volume']*1000
        df['lastPx'] = df.groupby('date')['lastPx'].ffill()
        df['size'] = df.groupby('date')['size'].transform(lambda x: x.fillna(0))
        #df['value'] = df['lastPx'] * df['size']
        df['value'] = df.groupby('date')['cumValue'].diff()
        df['value'] = df['value'].fillna(df['bfValue'])
        del df['bfValue']
        del df['cumValue']
        del df['value']
        
        ###Next, we create datetime, then bin the data to minutely before sending to signal calculation
        df_DateTime = pd.to_datetime(df.date.astype(str) + ' ' + df.time.astype(str), format="%Y-%m-%d %H%M%S%f")
        df['dt_index'] = df_DateTime
        df = df[~df.dt_index.duplicated(keep='last')]        
        
        binSize = '1min'
        
        ###Now, we bin the data to minutely        
        df_minutely = df.groupby(pd.Grouper(key='dt_index', freq=binSize, closed='right', label='right')).last()
        for i in range(1, 6):
            df_minutely.loc[:, 'SP{}'.format(str(i))] = df.groupby(pd.Grouper(key='dt_index', freq=binSize, closed='right', label='right'))['SP{}'.format(str(i))].last()
            df_minutely.loc[:, 'BP{}'.format(str(i))] = df.groupby(pd.Grouper(key='dt_index', freq=binSize, closed='right', label='right'))['BP{}'.format(str(i))].last()
            df_minutely.loc[:, 'SV{}'.format(str(i))] = df.groupby(pd.Grouper(key='dt_index', freq=binSize, closed='right', label='right'))['SV{}'.format(str(i))].last()
            df_minutely.loc[:, 'BV{}'.format(str(i))] = df.groupby(pd.Grouper(key='dt_index', freq=binSize, closed='right', label='right'))['BV{}'.format(str(i))].last()
        
        #do some cleaning
        df_minutely = df_minutely.between_time('09:00:00','13:25:00', inclusive = 'right')
        df_minutely['date'] = df_minutely.index.date
        df_minutely['ttime'] = df_minutely.index.time
        #df_minutely['time'].fillna(df_minutely['ttime'], inplace=True)
        df_minutely.fillna({'time': df_minutely['ttime']}, inplace=True)
        del df_minutely['ttime']
        #only keep trading days
        df_minutely = df_minutely[df_minutely['date'].astype(str).isin(tradingDays)]
            
        return df_minutely

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_verge(x, y):
    x = np.mean(x)
    y = np.mean(y)
    return np.sqrt(x ** 2 + y ** 2)

if __name__ == '__main__':
    ###Prepare common directory
    stockDataDir = 'E:\\Workbench\\UChicago\\UChicago_2025MMMLectures\\Assignments\\Assignment4\\references\\'
    
    ###Prepare stock list
    stock = "0050"

    cols = ["date","time","lastPx","size","volume","SP1","BP1","SV1","BV1","SP2","BP2","SV2","BV2","SP3","BP3","SV3","BV3","SP4","BP4","SV4","BV4","SP5","BP5","SV5","BV5"]

    tradingDays = ["2023-10-02","2023-10-03","2023-10-04","2023-10-05","2023-10-06","2023-10-11","2023-10-12","2023-10-13","2023-10-16","2023-10-17","2023-10-18","2023-10-19","2023-10-20","2023-10-23","2023-10-24","2023-10-25","2023-10-26","2023-10-27","2023-10-30","2023-10-31","2023-11-01","2023-11-02","2023-11-03","2023-11-06","2023-11-07","2023-11-08","2023-11-09","2023-11-10","2023-11-13","2023-11-14","2023-11-15","2023-11-16","2023-11-17","2023-11-20","2023-11-21","2023-11-22","2023-11-23","2023-11-24","2023-11-27","2023-11-28","2023-11-29","2023-11-30","2023-12-01","2023-12-04","2023-12-05","2023-12-06","2023-12-07","2023-12-08","2023-12-11","2023-12-12","2023-12-13","2023-12-14","2023-12-15","2023-12-18","2023-12-19","2023-12-20","2023-12-21","2023-12-22","2023-12-25","2023-12-26","2023-12-27","2023-12-28","2023-12-29","2024-01-02","2024-01-03","2024-01-04","2024-01-05","2024-01-08","2024-01-09","2024-01-10","2024-01-11","2024-01-12","2024-01-15","2024-01-16","2024-01-17","2024-01-18","2024-01-19","2024-01-22","2024-01-23","2024-01-24","2024-01-25","2024-01-26","2024-01-29","2024-01-30","2024-01-31","2024-02-01","2024-02-02","2024-02-15","2024-02-16","2024-02-19","2024-02-20","2024-02-21","2024-02-22","2024-02-23","2024-02-26","2024-02-27","2024-02-29","2024-03-01","2024-03-04","2024-03-05","2024-03-06","2024-03-07","2024-03-08","2024-03-11","2024-03-12","2024-03-13","2024-03-14","2024-03-15","2024-03-18","2024-03-19","2024-03-20","2024-03-21","2024-03-22","2024-03-25","2024-03-26","2024-03-27","2024-03-28","2024-03-29","2024-04-01","2024-04-02","2024-04-03","2024-04-08","2024-04-09","2024-04-10","2024-04-11","2024-04-12","2024-04-15","2024-04-16","2024-04-17","2024-04-18","2024-04-19","2024-04-22","2024-04-23","2024-04-24","2024-04-25","2024-04-26","2024-04-29","2024-04-30","2024-05-02","2024-05-03","2024-05-06","2024-05-07","2024-05-08","2024-05-09","2024-05-10","2024-05-13","2024-05-14","2024-05-15","2024-05-16","2024-05-17","2024-05-20","2024-05-21","2024-05-22","2024-05-23","2024-05-24","2024-05-27","2024-05-28","2024-05-29","2024-05-30","2024-05-31","2024-06-03","2024-06-04","2024-06-05","2024-06-06","2024-06-07","2024-06-11","2024-06-12","2024-06-13","2024-06-14","2024-06-17","2024-06-18","2024-06-19","2024-06-20","2024-06-21","2024-06-24","2024-06-25","2024-06-26","2024-06-27","2024-06-28","2024-07-01","2024-07-02","2024-07-03","2024-07-04","2024-07-05","2024-07-08","2024-07-09","2024-07-10","2024-07-11","2024-07-12","2024-07-15","2024-07-16","2024-07-17","2024-07-18","2024-07-19","2024-07-22","2024-07-23","2024-07-26","2024-07-29","2024-07-30","2024-07-31","2024-08-01","2024-08-02","2024-08-05","2024-08-06","2024-08-07","2024-08-08","2024-08-09","2024-08-12","2024-08-13","2024-08-14","2024-08-15","2024-08-16","2024-08-19","2024-08-20","2024-08-21","2024-08-22","2024-08-23","2024-08-26","2024-08-27","2024-08-28","2024-08-29","2024-08-30","2024-09-02","2024-09-03","2024-09-04","2024-09-05","2024-09-06","2024-09-09","2024-09-10","2024-09-11","2024-09-12","2024-09-13","2024-09-16","2024-09-18","2024-09-19","2024-09-20","2024-09-23","2024-09-24","2024-09-25","2024-09-26","2024-09-27","2024-09-30","2024-10-01","2024-10-02","2024-10-03","2024-10-04","2024-10-07","2024-10-08","2024-10-09","2024-10-11","2024-10-14","2024-10-15","2024-10-16","2024-10-17","2024-10-18","2024-10-21","2024-10-22","2024-10-23","2024-10-24","2024-10-25","2024-10-28","2024-10-29","2024-10-30","2024-10-31","2024-11-01","2024-11-04","2024-11-05","2024-11-06","2024-11-07","2024-11-08","2024-11-11","2024-11-12","2024-11-13","2024-11-14","2024-11-15","2024-11-18","2024-11-19","2024-11-20","2024-11-21","2024-11-22","2024-11-25","2024-11-26","2024-11-27","2024-11-28","2024-11-29","2024-12-02","2024-12-03","2024-12-04","2024-12-05","2024-12-06","2024-12-09","2024-12-10","2024-12-11","2024-12-12","2024-12-13","2024-12-16","2024-12-17","2024-12-18","2024-12-19","2024-12-20","2024-12-23","2024-12-24","2024-12-25","2024-12-26","2024-12-27","2024-12-30","2024-12-31"]

    print("Raw data loading and processing " + stock)
            
    ###load stock tick data (gzip)
    file1Path = stockDataDir + stock + '_md_202401_202401.csv.gz'
    file2Path = stockDataDir + stock + '_md_202402_202402.csv.gz'
    file3Path = stockDataDir + stock + '_md_202403_202403.csv.gz'
    df = pd.DataFrame()
    if os.path.exists(file1Path):
        df = pd.concat([df, pd.read_csv(file1Path, compression='gzip', usecols = cols)])
        print('Data 1 for ' + stock + ' loaded.')
    else:
        print('Skipping snapshots data ' + file1Path + ' for ' + stock + '.')
    if os.path.exists(file2Path):
        df = pd.concat([df, pd.read_csv(file2Path, compression='gzip', usecols = cols)])
        print('Data 2 for ' + stock + ' loaded.')
    else:
        print('Skipping snapshots data ' + file2Path + ' for ' + stock + '.')
    if os.path.exists(file3Path):
        df = pd.concat([df, pd.read_csv(file3Path, compression='gzip', usecols = cols)])
        print('Data 3 for ' + stock + ' loaded.')
    else:
        print('Skipping snapshots data ' + file3Path + ' for ' + stock + '.')

    if df.empty:
        print('No order snapshot data loaded. Skipping ' + stock)
        print("No raw data to process; exit.")
        
    else:    
        minutelyData = prepareMinutelyData(df, tradingDays)
        print("Minutely data generated.")
    
    projdata = []
    columns = ['date', 'time', 'lastPx', 'size', 'volume',
               'SP5', 'SP4', 'SP3', 'SP2', 'SP1',
               'BP1', 'BP2', 'BP3', 'BP4', 'BP5',
               'SV5', 'SV4', 'SV3', 'SV2', 'SV1',
               'BV1', 'BV2', 'BV3', 'BV4', 'BV5']
    
    for x in minutelyData.groupby('date'):
        if x[1].shape[0] == 265:
            projdata.append(x[1].values)

    projdata = np.array(projdata)    
    
    #normalization
    X = projdata[:,:,5:].astype(float)
    
    X[:,:,-10:] = np.log(1 + X[:,:,-10:])
    X_mean = X.mean(axis=1)
    X_std = X.std(axis=1)
    
    X = np.transpose((np.transpose(X, (1,0,2)) - X_mean) / (2 * X_std), (1,0,2))
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    #set up
    set_seed(307)
    
    generator = Generator()
    discriminator = Discriminator()
    
    #batch size
    batch_size = 50
    dataset = MyDataset(torch.tensor(X, dtype=torch.float32))
    
    #training, validation, testing
    test_dataset = dataset[0:]
    
    #dataloaders
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    
    #load the model
    generator = torch.load(stock+'_generator1.pth', weights_only=False)
    discriminator = torch.load(stock+'_discriminator1.pth', weights_only=False)
    print("Model loaded.")    
    
    #apply test data and check the abnormal data
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    test_data = projdata[:, :, :]
    
    # daily returns
    price = test_data[:,:,[0,2,-1]]
    price = pd.DataFrame(price.reshape((-1, 3)))
    price.columns = ['date', 'price', 'index']
    def get_return(x):
        return x.iloc[-1] / x.iloc[0] - 1
    ret = price.groupby(['date', 'index']).apply(get_return)
    ret.index = ret.index.swaplevel()
    ret = ret.rename(columns={'price': 'realRtn'})

    # compare "abnormal" quantifiers
    discriminator.eval()
    dis_index_list = []
    dis_ret_list = []
    dis_tomoret_list = []
    for i, data in enumerate(test_dataloader):
        if discriminator(data) <= 0.5:
            #counter += 1
            index = test_data[i, 0, -1]
            
            today_return = test_data[i,-1,2] / test_data[i,0,2] - 1
            dis_ret_list.append(today_return)
            dis_index_list.append(index)
            
            j = i + 1
            while j < test_data.shape[0] and test_data[j, 0, -1] != index:
                j += 1
            if j == test_data.shape[0]: j -= 1
            tomorrow_return = test_data[j,-1,2] / test_data[j,0,2] - 1
            dis_tomoret_list.append(tomorrow_return)    
        
        genLOBData = pd.DataFrame(generator(data).detach().numpy()[0])
        
    dis_ret = pd.DataFrame()
    dis_ret.index = dis_index_list
    dis_ret['return'] = dis_ret_list
    dis_ret['tomorrow_return'] = dis_tomoret_list
    
    ax1 = sns.kdeplot(dis_ret['return'].values, linestyle='--', fill=True, label='discrimatedRtn')
    ax2 = sns.kdeplot(ret, label='realRtn')
    plt.legend()
    plt.show()
    plt.savefig(stock + '_return.png')