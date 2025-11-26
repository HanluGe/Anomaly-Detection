# -*- coding: utf-8 -*-
"""
Created on Wed October 8 09:40:23 2025

@author: hongs; adapted from the orignal copy by Mr. GUAN Chenjiong, 2025.
"""

import os

# ===== 防止 Windows 下 OpenMP 冲突导致 kernel 直接崩溃 =====
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  
os.environ["OMP_NUM_THREADS"] = "1"         
os.environ["MKL_NUM_THREADS"] = "1"         

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.lay1 = nn.GRU(20, 40, num_layers=1, batch_first=True)
        self.lay2 = nn.Sequential(nn.Linear(40, 40), nn.LeakyReLU(0.01), nn.Linear(40, 40))
        self.lay3 = nn.GRU(40, 40, num_layers=1, batch_first=True)
        self.lay4 = nn.Sequential(nn.Linear(40, 40), nn.LeakyReLU(0.01), nn.Linear(40, 40))
        self.lay5 = nn.GRU(40, 19, num_layers=1, batch_first=True)  # note: one layer should have less than 20 nodes to avoid data repeatation
        self.lay6 = nn.Sequential(nn.Linear(19, 40), nn.LeakyReLU(0.01), nn.Linear(40, 40))
        self.lay7 = nn.GRU(40, 20, num_layers=1, batch_first=True)
        self.lay8 = nn.Sequential(nn.Linear(40, 40), nn.LeakyReLU(0.01), nn.Linear(40, 20))


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


def train_lob_gan(
    stock: str,
    lr_g: float = 0.00375,
    lr_d: float = 0.001,
    batch_size: int = 50,
    seed: int = 307,
):
    set_seed(seed)

    cols = ["date","time","lastPx","size","volume","SP1","BP1","SV1","BV1","SP2","BP2","SV2","BV2","SP3","BP3","SV3","BV3","SP4","BP4","SV4","BV4","SP5","BP5","SV5","BV5"]

    tradingDays = ["2023-10-02","2023-10-03","2023-10-04","2023-10-05","2023-10-06","2023-10-11","2023-10-12","2023-10-13","2023-10-16","2023-10-17","2023-10-18","2023-10-19","2023-10-20","2023-10-23","2023-10-24","2023-10-25","2023-10-26","2023-10-27","2023-10-30","2023-10-31","2023-11-01","2023-11-02","2023-11-03","2023-11-06","2023-11-07","2023-11-08","2023-11-09","2023-11-10","2023-11-13","2023-11-14","2023-11-15","2023-11-16","2023-11-17","2023-11-20","2023-11-21","2023-11-22","2023-11-23","2023-11-24","2023-11-27","2023-11-28","2023-11-29","2023-11-30","2023-12-01","2023-12-04","2023-12-05","2023-12-06","2023-12-07","2023-12-08","2023-12-11","2023-12-12","2023-12-13","2023-12-14","2023-12-15","2023-12-18","2023-12-19","2023-12-20","2023-12-21","2023-12-22","2023-12-25","2023-12-26","2023-12-27","2023-12-28","2023-12-29","2024-01-02","2024-01-03","2024-01-04","2024-01-05","2024-01-08","2024-01-09","2024-01-10","2024-01-11","2024-01-12","2024-01-15","2024-01-16","2024-01-17","2024-01-18","2024-01-19","2024-01-22","2024-01-23","2024-01-24","2024-01-25","2024-01-26","2024-01-29","2024-01-30","2024-01-31","2024-02-01","2024-02-02","2024-02-15","2024-02-16","2024-02-19","2024-02-20","2024-02-21","2024-02-22","2024-02-23","2024-02-26","2024-02-27","2024-02-29","2024-03-01","2024-03-04","2024-03-05","2024-03-06","2024-03-07","2024-03-08","2024-03-11","2024-03-12","2024-03-13","2024-03-14","2024-03-15","2024-03-18","2024-03-19","2024-03-20","2024-03-21","2024-03-22","2024-03-25","2024-03-26","2024-03-27","2024-03-28","2024-03-29","2024-04-01","2024-04-02","2024-04-03","2024-04-08","2024-04-09","2024-04-10","2024-04-11","2024-04-12","2024-04-15","2024-04-16","2024-04-17","2024-04-18","2024-04-19","2024-04-22","2024-04-23","2024-04-24","2024-04-25","2024-04-26","2024-04-29","2024-04-30","2024-05-02","2024-05-03","2024-05-06","2024-05-07","2024-05-08","2024-05-09","2024-05-10","2024-05-13","2024-05-14","2024-05-15","2024-05-16","2024-05-17","2024-05-20","2024-05-21","2024-05-22","2024-05-23","2024-05-24","2024-05-27","2024-05-28","2024-05-29","2024-05-30","2024-05-31","2024-06-03","2024-06-04","2024-06-05","2024-06-06","2024-06-07","2024-06-11","2024-06-12","2024-06-13","2024-06-14","2024-06-17","2024-06-18","2024-06-19","2024-06-20","2024-06-21","2024-06-24","2024-06-25","2024-06-26","2024-06-27","2024-06-28","2024-07-01","2024-07-02","2024-07-03","2024-07-04","2024-07-05","2024-07-08","2024-07-09","2024-07-10","2024-07-11","2024-07-12","2024-07-15","2024-07-16","2024-07-17","2024-07-18","2024-07-19","2024-07-22","2024-07-23","2024-07-26","2024-07-29","2024-07-30","2024-07-31","2024-08-01","2024-08-02","2024-08-05","2024-08-06","2024-08-07","2024-08-08","2024-08-09","2024-08-12","2024-08-13","2024-08-14","2024-08-15","2024-08-16","2024-08-19","2024-08-20","2024-08-21","2024-08-22","2024-08-23","2024-08-26","2024-08-27","2024-08-28","2024-08-29","2024-08-30","2024-09-02","2024-09-03","2024-09-04","2024-09-05","2024-09-06","2024-09-09","2024-09-10","2024-09-11","2024-09-12","2024-09-13","2024-09-16","2024-09-18","2024-09-19","2024-09-20","2024-09-23","2024-09-24","2024-09-25","2024-09-26","2024-09-27","2024-09-30","2024-10-01","2024-10-02","2024-10-03","2024-10-04","2024-10-07","2024-10-08","2024-10-09","2024-10-11","2024-10-14","2024-10-15","2024-10-16","2024-10-17","2024-10-18","2024-10-21","2024-10-22","2024-10-23","2024-10-24","2024-10-25","2024-10-28","2024-10-29","2024-10-30","2024-10-31","2024-11-01","2024-11-04","2024-11-05","2024-11-06","2024-11-07","2024-11-08","2024-11-11","2024-11-12","2024-11-13","2024-11-14","2024-11-15","2024-11-18","2024-11-19","2024-11-20","2024-11-21","2024-11-22","2024-11-25","2024-11-26","2024-11-27","2024-11-28","2024-11-29","2024-12-02","2024-12-03","2024-12-04","2024-12-05","2024-12-06","2024-12-09","2024-12-10","2024-12-11","2024-12-12","2024-12-13","2024-12-16","2024-12-17","2024-12-18","2024-12-19","2024-12-20","2024-12-23","2024-12-24","2024-12-25","2024-12-26","2024-12-27","2024-12-30","2024-12-31"]

    print("Raw data loading and processing " + stock)

    file1Path = os.path.join(DATA_DIR, stock + '_md_202310_202310.csv.gz')
    file2Path = os.path.join(DATA_DIR, stock + '_md_202311_202311.csv.gz')
    file3Path = os.path.join(DATA_DIR, stock + '_md_202312_202312.csv.gz')

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
        return None
    
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
    set_seed(seed)
    
    generator = Generator()
    discriminator = Discriminator()
    
    #params
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.99, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.99, 0.999))
    
    #batch size
    batch_size = batch_size
    dataset = MyDataset(torch.tensor(X, dtype=torch.float32))
    
    #training, validation, testing
    train_size = int(0.8 * len(dataset))
    eval_size = int(0.2 * len(dataset))
    train_dataset, eval_dataset = random_split(dataset[:train_size+eval_size], [train_size, eval_size])
    
    #dataloaders
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=True)
    
    #loss functions
    gen_loss = torch.nn.BCELoss()
    loss_function = torch.nn.MSELoss()    
        
    epochs = 200
    
    #storage of loss data
    train_g_loss = [] #
    train_d_loss = [] #
    eval_g_loss = [] #
    eval_d_loss = [] #
    
    train_verge = [] #early stopping
    eval_verge = [] #early stopping

    print(f"Start training {stock}: epochs={epochs}, batch_size={batch_size}, lr_g={lr_g}, lr_d={lr_d}")

    #training starts here
    ## 在 train_dataloader 上训练（有反向传播 + 更新参数）
    for epoch in range(epochs):
        for i, data in enumerate(train_dataloader):
    
            #real vs noise
            real = torch.ones(data.size(0), 1)
            fake = torch.zeros(data.size(0), 1)
    
            #train the generator
            generator.train()
            optimizer_G.zero_grad()
    
            gen = generator(data)
    
            d_data = data[:,1:,:] - data[:,:-1,:]
            d_gen = gen[:,1:,:] - gen[:,:-1,:]
            dd_data = d_data[:,1:,:] - d_data[:,:-1,:]
            dd_gen = d_gen[:,1:,:] - d_gen[:,:-1,:]
    
            g_loss = loss_function(discriminator(gen), real) + \
                     loss_function(torch.mean(torch.abs(data), axis=1), torch.mean(torch.abs(gen), axis=1)) + \
                     loss_function(torch.mean(data, axis=1), torch.mean(gen, axis=1)) + \
                     loss_function(torch.mean(data ** 2, axis=1), torch.mean(gen ** 2, axis=1)) + \
                     loss_function(torch.mean(data ** 3, axis=1), torch.mean(gen ** 3, axis=1)) + \
                     loss_function(torch.mean(torch.abs(d_data), axis=1), torch.mean(torch.abs(d_gen), axis=1)) + \
                     loss_function(torch.mean(d_data, axis=1), torch.mean(d_gen, axis=1)) + \
                     loss_function(torch.mean(d_data ** 2, axis=1), torch.mean(d_gen ** 2, axis=1)) +\
                     loss_function(torch.mean(d_data ** 3, axis=1), torch.mean(d_gen ** 3, axis=1)) +\
                     loss_function(torch.mean(torch.abs(dd_data), axis=1), torch.mean(torch.abs(dd_gen), axis=1)) + \
                     loss_function(torch.mean(dd_data, axis=1), torch.mean(dd_gen, axis=1)) + \
                     loss_function(torch.mean(dd_data ** 2, axis=1), torch.mean(dd_gen ** 2, axis=1)) +\
                     loss_function(torch.mean(dd_data ** 3, axis=1), torch.mean(dd_gen ** 3, axis=1))
    
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 0.3) #clipping the gradient
            optimizer_G.step()
    
            #train the discriminator
            discriminator.train()
            optimizer_D.zero_grad()
    
            real_loss = loss_function(discriminator(data), real)
            fake_loss = loss_function(discriminator(gen.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
    
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 0.1) #clipping the gradient
            optimizer_D.step()
    
            train_g_loss.append(g_loss.item())
            train_d_loss.append(d_loss.item())
    
            if i % 10 == 0:
                print("[Epoch %d/%d][Batch %d/%d][D train loss: %f][G train loss: %f]" % (epoch+1, epochs, i+1, len(train_dataloader),
                                                                             d_loss.item(), g_loss.item()))
    
        #validation data set

        g_loss_total = 0
        d_loss_total = 0
        for i, data in enumerate(eval_dataloader): # 在 eval_dataloader 上验证（只算 loss，不更新参数，用来监控 & early stopping）
    
            #real vs. noise
            real = torch.ones(data.size(0), 1)
            fake = torch.zeros(data.size(0), 1)
    
            #evaluate the generator
            generator.eval()
    
            gen = generator(data)
    
            d_data = data[:,1:,:] - data[:,:-1,:]
            d_gen = gen[:,1:,:] - gen[:,:-1,:]
            dd_data = d_data[:,1:,:] - d_data[:,:-1,:]
            dd_gen = d_gen[:,1:,:] - d_gen[:,:-1,:]
    
            g_loss = loss_function(discriminator(gen), real) + \
                     loss_function(torch.mean(torch.abs(data), axis=1), torch.mean(torch.abs(gen), axis=1)) + \
                     loss_function(torch.mean(data, axis=1), torch.mean(gen, axis=1)) + \
                     loss_function(torch.mean(data ** 2, axis=1), torch.mean(gen ** 2, axis=1)) + \
                     loss_function(torch.mean(data ** 3, axis=1), torch.mean(gen ** 3, axis=1)) + \
                     loss_function(torch.mean(torch.abs(d_data), axis=1), torch.mean(torch.abs(d_gen), axis=1)) + \
                     loss_function(torch.mean(d_data, axis=1), torch.mean(d_gen, axis=1)) + \
                     loss_function(torch.mean(d_data ** 2, axis=1), torch.mean(d_gen ** 2, axis=1)) +\
                     loss_function(torch.mean(d_data ** 3, axis=1), torch.mean(d_gen ** 3, axis=1)) +\
                     loss_function(torch.mean(torch.abs(dd_data), axis=1), torch.mean(torch.abs(dd_gen), axis=1)) + \
                     loss_function(torch.mean(dd_data, axis=1), torch.mean(dd_gen, axis=1)) + \
                     loss_function(torch.mean(dd_data ** 2, axis=1), torch.mean(dd_gen ** 2, axis=1)) +\
                     loss_function(torch.mean(dd_data ** 3, axis=1), torch.mean(dd_gen ** 3, axis=1))
            g_loss_total += g_loss
    
            #evaluate the discriminator
            discriminator.eval()
    
            real_loss = loss_function(discriminator(data), real)
            fake_loss = loss_function(discriminator(gen.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss_total += d_loss
    
            eval_g_loss.append(g_loss.item())
            eval_d_loss.append(d_loss.item())
    
            
        print("[Epoch %d/%d][Batch %d/%d][D eval loss: %f][G eval loss: %f]" % (epoch+1, epochs, i+1, len(eval_dataloader),
                                                                               d_loss_total.item()/len(eval_dataloader),
                                                                                g_loss_total.item()/len(eval_dataloader)))
    
        train_verge.append(get_verge(train_g_loss[-len(train_dataloader):], train_d_loss[-len(train_dataloader):]))    
        eval_verge.append(get_verge(eval_g_loss[-len(eval_dataloader):], eval_d_loss[-len(eval_dataloader):]))
    
        if epoch >= 5:
            #early stop
            if train_verge[-3] > train_verge[-2] and train_verge[-2] > train_verge[-1] and eval_verge[-3] < eval_verge[-2] and eval_verge[-2] < eval_verge[-1]:
                print("Early stopping.")
                break

    train_csv_path = os.path.join(RESULTS_DIR, stock+'_train_g_d.csv')
    eval_csv_path  = os.path.join(RESULTS_DIR, stock+'_eval_g_d.csv')
    pd.DataFrame([train_g_loss, train_d_loss], index=['train_g', 'train_d']).to_csv(train_csv_path)
    pd.DataFrame([eval_g_loss, eval_d_loss], index=['eval_g', 'eval_d']).to_csv(eval_csv_path)

    try:
        png_path = os.path.join(RESULTS_DIR, f"{stock}_loss_curve.png")
        plt.figure(figsize=(10, 6))
        plt.plot(train_g_loss, label='Train G loss')
        plt.plot(eval_g_loss, label='Eval G loss')
        plt.plot(train_d_loss, label='Train D loss')
        plt.plot(eval_d_loss, label='Eval D loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'GAN Training Loss - {stock}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()
        print("Saved loss curve to", png_path)
    except Exception as e:
        print("Warning: failed to save loss curve:", repr(e))

    try:
        gen_path  = os.path.join(RESULTS_DIR, stock+'_generator1.pth')
        disc_path = os.path.join(RESULTS_DIR, stock+'_discriminator1.pth')
        torch.save(generator, gen_path)
        torch.save(discriminator, disc_path)
        print("Saved models to", gen_path, "and", disc_path)
    except Exception as e:
        print("Warning: failed to save models:", repr(e))

    print("Done training LOB_GAN for stock " + stock)
    print("Results saved to:", RESULTS_DIR)

    return {
        "train_g_loss": train_g_loss,
        "train_d_loss": train_d_loss,
        "eval_g_loss": eval_g_loss,
        "eval_d_loss": eval_d_loss
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", type=str, required=True)
    parser.add_argument("--lr_g", type=float, default=0.00375)
    parser.add_argument("--lr_d", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=307)
    args = parser.parse_args()

    train_lob_gan(
        stock=args.stock,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        batch_size=args.batch_size,
        seed=args.seed,
    )
