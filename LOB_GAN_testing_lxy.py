# -*- coding: utf-8 -*-
"""
Q2 用的测试脚本：
- 读取 2024-01 ~ 2024-03 的分钟数据；
- 加载已经训练好的 raw / adjusted 模型；
- 用 Discriminator 挑出 “abnormal” 的样本；
- 返回：dis_ret（abnormal 日的当日 & 次日收益）和 ret（全体样本真实收益分布）。
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import seaborn as sns
import matplotlib.pyplot as plt

# ------------ 路径设置（和你同学的 train.py 一致） ------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")      # 所有 *_md_2024xx_2024xx.csv.gz 在这里
RESULTS_DIR = os.path.join(BASE_DIR, "Results") # 训练好的 pth / csv / png 在这里

os.makedirs(RESULTS_DIR, exist_ok=True)


# ------------ 模型结构（和原始训练脚本一致；这里只是占位，真正 forward 来自 torch.load） ------------

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.lay1 = nn.GRU(20, 40, num_layers=1, batch_first=True)
        self.lay2 = nn.Sequential(nn.Linear(40, 40), nn.LeakyReLU(0.01), nn.Linear(40, 40))
        self.lay3 = nn.GRU(40, 40, num_layers=1, batch_first=True)
        self.lay4 = nn.Sequential(nn.Linear(40, 40), nn.LeakyReLU(0.01), nn.Linear(40, 40))
        self.lay5 = nn.GRU(40, 19, num_layers=1, batch_first=True)
        self.lay6 = nn.Sequential(nn.Linear(19, 40), nn.LeakyReLU(0.01), nn.Linear(40, 40))
        self.lay7 = nn.GRU(40, 20, num_layers=1, batch_first=True)
        self.lay8 = nn.Sequential(nn.Linear(40, 40), nn.LeakyReLU(0.01), nn.Linear(40, 20))

    def forward(self, x):
        # 注意：真正用的 forward 来自 torch.load 出来的模型，这个 class 只是为了兼容
        y, _ = self.lay1(x)
        z = self.lay2(y)
        u, _ = self.lay3(z)
        v = self.lay4(u)
        w, _ = self.lay5(v)
        o = self.lay6(w)
        p, _ = self.lay7(o)
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


def prepareMinutelyData(df: pd.DataFrame, tradingDays: list):
    if df.empty:
        return None

    df['bfValue'] = df['lastPx'] * df['size']
    df['bfValue'] = df['bfValue'].ffill()
    df['cumValue'] = df.groupby('date')['bfValue'].cumsum()
    df = df[df['SP1'] > 0]
    df = df[df['BP1'] > 0]
    df = df[df['SP1'] - df['BP1'] > 0]
    for i in range(1, 6):
        df[f'SP{i}'] = df[f'SP{i}'] / 100
        df[f'BP{i}'] = df[f'BP{i}'] / 100
        df[f'SV{i}'] = df[f'SV{i}'] * 1000
        df[f'BV{i}'] = df[f'BV{i}'] * 1000
    df['lastPx'] = df['lastPx'] / 100
    df['size'] = df['size'] * 1000
    df['volume'] = df['volume'] * 1000
    df['lastPx'] = df.groupby('date')['lastPx'].ffill()
    df['size'] = df.groupby('date')['size'].transform(lambda x: x.fillna(0))
    df['value'] = df.groupby('date')['cumValue'].diff()
    df['value'] = df['value'].fillna(df['bfValue'])
    del df['bfValue'], df['cumValue'], df['value']

    df_DateTime = pd.to_datetime(
        df.date.astype(str) + ' ' + df.time.astype(str),
        format="%Y-%m-%d %H%M%S%f"
    )
    df['dt_index'] = df_DateTime
    df = df[~df.dt_index.duplicated(keep='last')]

    binSize = '1min'

    df_minutely = df.groupby(
        pd.Grouper(key='dt_index', freq=binSize, closed='right', label='right')
    ).last()
    for i in range(1, 6):
        grp = df.groupby(
            pd.Grouper(key='dt_index', freq=binSize, closed='right', label='right')
        )
        df_minutely.loc[:, f'SP{i}'] = grp[f'SP{i}'].last()
        df_minutely.loc[:, f'BP{i}'] = grp[f'BP{i}'].last()
        df_minutely.loc[:, f'SV{i}'] = grp[f'SV{i}'].last()
        df_minutely.loc[:, f'BV{i}'] = grp[f'BV{i}'].last()

    df_minutely = df_minutely.between_time('09:00:00', '13:25:00', inclusive='right')
    df_minutely['date'] = df_minutely.index.date
    df_minutely['ttime'] = df_minutely.index.time
    df_minutely.fillna({'time': df_minutely['ttime']}, inplace=True)
    del df_minutely['ttime']

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



def run_anomaly_detection(
    stock_code: str,
    version: str = "raw",     # "raw" 或 "adjusted"
    batch_size: int = 1,
    seed: int = 307,
    save_fig: bool = True
):
    """
    用已经训练好的 GAN 模型，对 2024-01~03 的分钟数据做 abnormal 检测。

    参数：
      stock_code: "0050" / "0056" / "2330" 等
      version   : "raw" or "adjusted"，决定加载哪个 pth
      batch_size: DataLoader 的 batch_size（这里一般用 1 就好）
      seed      : 随机种子
      save_fig  : 是否把 KDE 图保存为 png

    返回：
      dis_ret: DataFrame，index 为日期索引，列为 ['return', 'tomorrow_return']
      ret    : Series / DataFrame，真实日收益分布（保持原代码习惯）
    """
    set_seed(seed)

    # ==== 1. 读 2024-01~03 数据 ====
    cols = [
        "date", "time", "lastPx", "size", "volume",
        "SP1", "BP1", "SV1", "BV1",
        "SP2", "BP2", "SV2", "BV2",
        "SP3", "BP3", "SV3", "BV3",
        "SP4", "BP4", "SV4", "BV4",
        "SP5", "BP5", "SV5", "BV5"
    ]

    # 这里的 tradingDays 列表用你原 test.py 里的那一大串就行，这里不重复贴了
    tradingDays = [
        "2023-10-02","2023-10-03","2023-10-04","2023-10-05","2023-10-06",
        # ... 中间略，与 train.py/test.py 完全一致 ...
        "2024-12-30","2024-12-31"
    ]

    print("Raw data loading and processing " + stock_code)

    file1Path = os.path.join(DATA_DIR, stock_code + '_md_202401_202401.csv.gz')
    file2Path = os.path.join(DATA_DIR, stock_code + '_md_202402_202402.csv.gz')
    file3Path = os.path.join(DATA_DIR, stock_code + '_md_202403_202403.csv.gz')

    df = pd.DataFrame()
    if os.path.exists(file1Path):
        df = pd.concat([df, pd.read_csv(file1Path, compression='gzip', usecols=cols)])
        print('Data 1 for ' + stock_code + ' loaded.')
    else:
        print('Skipping snapshots data ' + file1Path + ' for ' + stock_code + '.')
    if os.path.exists(file2Path):
        df = pd.concat([df, pd.read_csv(file2Path, compression='gzip', usecols=cols)])
        print('Data 2 for ' + stock_code + ' loaded.')
    else:
        print('Skipping snapshots data ' + file2Path + ' for ' + stock_code + '.')
    if os.path.exists(file3Path):
        df = pd.concat([df, pd.read_csv(file3Path, compression='gzip', usecols=cols)])
        print('Data 3 for ' + stock_code + ' loaded.')
    else:
        print('Skipping snapshots data ' + file3Path + ' for ' + stock_code + '.')

    if df.empty:
        print('No order snapshot data loaded. Skipping ' + stock_code)
        print("No raw data to process; exit.")
        return None, None

    # ==== 2. 分钟数据 & 归一化 ====
    minutelyData = prepareMinutelyData(df, tradingDays)
    print("Minutely data generated.")

    projdata = []
    for x in minutelyData.groupby('date'):
        if x[1].shape[0] == 265:
            projdata.append(x[1].values)
    projdata = np.array(projdata)

    X = projdata[:, :, 5:].astype(float)
    X[:, :, -10:] = np.log(1 + X[:, :, -10:])
    X_mean = X.mean(axis=1)
    X_std = X.std(axis=1)

    X = np.transpose((np.transpose(X, (1, 0, 2)) - X_mean) / (2 * X_std), (1, 0, 2))
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    dataset = MyDataset(torch.tensor(X, dtype=torch.float32))
    test_dataset = dataset[:]   # 全部用于测试
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # ==== 3. 加载已经训练好的模型 ====
    if version == "raw":
        gen_file = f"{stock_code}_generator1.pth"
        disc_file = f"{stock_code}_discriminator1.pth"
    elif version == "adjusted":
        gen_file = f"{stock_code}_generator1_adjusted.pth"
        disc_file = f"{stock_code}_discriminator1_adjusted.pth"
    else:
        raise ValueError("version must be 'raw' or 'adjusted'")

    gen_path = os.path.join(RESULTS_DIR, gen_file)
    disc_path = os.path.join(RESULTS_DIR, disc_file)

    if not os.path.exists(gen_path) or not os.path.exists(disc_path):
        print(f"Model files not found: {gen_path} or {disc_path}")
        return None, None

    generator = torch.load(gen_path, weights_only=False)
    discriminator = torch.load(disc_path, weights_only=False)
    print(f"Loaded models from {gen_path} and {disc_path}")

    # ==== 4. 计算 daily return & abnormal 集合 ====
    test_data = projdata[:, :, :]
    price = test_data[:, :, [0, 2, -1]]  # [date, price, index]
    price = pd.DataFrame(price.reshape((-1, 3)))
    price.columns = ['date', 'price', 'index']

    def get_return(x):
        return x.iloc[-1] / x.iloc[0] - 1

    ret = price.groupby(['date', 'index']).apply(get_return)
    # 这里 ret 的结构有点奇怪，但是保持和原代码一致，方便你后面对比
    # ret.index = ret.index.swaplevel()  # 如果你原来就有这句，可以加上
    # ret = ret.rename(columns={'price': 'realRtn'})  # 原代码想 rename，但实际 ret 是 Series

    discriminator.eval()
    dis_index_list = []
    dis_ret_list = []
    dis_tomoret_list = []

    for i, data in enumerate(test_dataloader):
        with torch.no_grad():
            score = discriminator(data)
        if score <= 0.5:
            index = test_data[i, 0, -1]

            today_return = test_data[i, -1, 2] / test_data[i, 0, 2] - 1
            dis_ret_list.append(today_return)
            dis_index_list.append(index)

            j = i + 1
            while j < test_data.shape[0] and test_data[j, 0, -1] != index:
                j += 1
            if j == test_data.shape[0]:
                j -= 1
            tomorrow_return = test_data[j, -1, 2] / test_data[j, 0, 2] - 1
            dis_tomoret_list.append(tomorrow_return)

            # 如果你想看生成的 LOB，可以保留这句
            genLOBData = pd.DataFrame(generator(data).detach().numpy()[0])

    dis_ret = pd.DataFrame()
    dis_ret.index = dis_index_list
    dis_ret['return'] = dis_ret_list
    dis_ret['tomorrow_return'] = dis_tomoret_list

    # ==== 5. 画 KDE 对比图 ====
    plt.figure(figsize=(8, 5))
    sns.kdeplot(dis_ret['return'].values, linestyle='--', fill=True, label='discriminatedRtn')
    try:
        sns.kdeplot(ret, label='realRtn')
    except Exception:
        pass
    plt.legend()
    plt.title(f"KDE of daily returns - {stock_code} ({version})")
    if save_fig:
        fig_path = os.path.join(RESULTS_DIR, f"{stock_code}_return_{version}.png")
        plt.savefig(fig_path, dpi=200)
        print("Saved KDE figure to:", fig_path)
    plt.show()

    return dis_ret, ret


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", type=str, required=True)
    parser.add_argument("--version", type=str, default="raw", choices=["raw", "adjusted"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=307)
    parser.add_argument("--no_save_fig", action="store_true")
    args = parser.parse_args()

    run_anomaly_detection(
        stock_code=args.stock,
        version=args.version,
        batch_size=args.batch_size,
        seed=args.seed,
        save_fig=(not args.no_save_fig),
    )
