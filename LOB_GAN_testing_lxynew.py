# -*- coding: utf-8 -*-
"""
LOB_GAN_testing.py
用于 HW4 Q2:
- 使用已经训练好的 Discriminator 在测试集（2024-01 ~ 2024-03）上挑出 "abnormal" 日度 LOB 序列；
- 将该日的所有分钟级 snapshot 归类为 abnormal / normal；
- 对若干 microstructure 变量做描述性统计 + KS 检验。
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp, skew, kurtosis  # KS 检验 + 偏度/峰度


# --------------------------------------------------------
# 目录设置（和你同学的 train.py 一致）
# --------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# --------------------------------------------------------
# Model skeletons (结构要和训练时一致，方便 torch.load)
# 真正的 forward 用的是 load 出来的模型参数
# --------------------------------------------------------
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.lay1 = nn.GRU(20, 40, num_layers=1, batch_first=True)
        self.lay2 = nn.Sequential(
            nn.Linear(40, 40),
            nn.LeakyReLU(0.01),
            nn.Linear(40, 40)
        )
        self.lay3 = nn.GRU(40, 40, num_layers=1, batch_first=True)
        self.lay4 = nn.Sequential(
            nn.Linear(40, 40),
            nn.LeakyReLU(0.01),
            nn.Linear(40, 40)
        )
        self.lay5 = nn.GRU(40, 19, num_layers=1, batch_first=True)
        self.lay6 = nn.Sequential(
            nn.Linear(19, 40),
            nn.LeakyReLU(0.01),
            nn.Linear(40, 40)
        )
        self.lay7 = nn.GRU(40, 20, num_layers=1, batch_first=True)
        self.lay8 = nn.Sequential(
            nn.Linear(40, 40),
            nn.LeakyReLU(0.01),
            nn.Linear(40, 20)
        )

    def forward(self, x):
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
        self.lay2 = nn.Sequential(
            nn.Linear(40, 40),
            nn.LeakyReLU(0.01),
            nn.Linear(40, 40)
        )
        self.lay3 = nn.GRU(40, 40, num_layers=1, batch_first=True)
        self.lay4 = nn.Sequential(
            nn.Linear(40, 40),
            nn.LeakyReLU(0.01),
            nn.Linear(40, 40)
        )
        self.lay5 = nn.GRU(40, 40, num_layers=1, batch_first=True)
        self.lay6 = nn.Sequential(
            nn.Linear(40, 40),
            nn.LeakyReLU(0.01),
            nn.Linear(40, 40)
        )
        self.lay7 = nn.GRU(40, 40, num_layers=1, batch_first=True)
        self.lay8 = nn.Sequential(
            nn.Linear(40, 40),
            nn.LeakyReLU(0.01),
            nn.Linear(40, 1)
        )

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


# --------------------------------------------------------
# Data processing
# --------------------------------------------------------
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
    grouped = df.groupby(
        pd.Grouper(key='dt_index', freq=binSize, closed='right', label='right')
    )
    df_minutely = grouped.last()
    for i in range(1, 6):
        df_minutely.loc[:, f'SP{i}'] = grouped[f'SP{i}'].last()
        df_minutely.loc[:, f'BP{i}'] = grouped[f'BP{i}'].last()
        df_minutely.loc[:, f'SV{i}'] = grouped[f'SV{i}'].last()
        df_minutely.loc[:, f'BV{i}'] = grouped[f'BV{i}'].last()

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


# --------------------------------------------------------
# 一点点工具函数
# --------------------------------------------------------
# def _clean_array(arr):
#     arr = np.asarray(arr).ravel()
#     mask = np.isfinite(arr)
#     return arr[mask]
def _clean_array(arr):
    """
    把输入尽最大可能转成 float 一维数组：
      - 先展平
      - 用 pandas.to_numeric 强制转成数值（出错的统统变 NaN）
      - 再用 np.isfinite 去掉 NaN / inf
    这样就算原始是 dtype=object、里面混 string / list 也不会再在 isfinite 这里报错。
    """
    # 展平为一维，统一先当 object
    arr = np.array(arr, dtype=object).ravel()

    # 变成 Series 再转数值，errors="coerce" 会把非数值变成 NaN
    s = pd.Series(arr, dtype="object")
    s_num = pd.to_numeric(s, errors="coerce")

    # 转成 float64 的 numpy 数组
    arr_f = s_num.to_numpy(dtype="float64")

    # 去掉 NaN / inf
    mask = np.isfinite(arr_f)
    return arr_f[mask]



def _describe_array(arr):
    arr = _clean_array(arr)
    if len(arr) == 0:
        return {"n": 0, "mean": np.nan, "var": np.nan, "skew": np.nan, "kurtosis": np.nan}
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "var": float(np.var(arr)),
        "skew": float(skew(arr, bias=False)),
        "kurtosis": float(kurtosis(arr, fisher=False, bias=False)),  # 正态=3
    }


# --------------------------------------------------------
# 核心函数：做 abnormal / normal 划分 + 统计 + KS test
# --------------------------------------------------------
def analyze_microstructure(
    stock_code: str,
    version: str = "raw",      # "raw" or "adjusted"
    threshold: float = 0.5,    # Discriminator <= threshold -> abnormal
    batch_size: int = 1,
    seed: int = 307,
    save_kde_fig: bool = False
):
    """
    对某一只股票，使用训练好的 Discriminator 划分 abnormal / normal，
    并对 microstructure 变量做统计 + KS test。

    返回：
      stats_df: 每个变量在 abnormal / normal 集合的 (n, mean, var, skew, kurtosis)
      ks_df   : 每个变量的 KS 统计量和 p-value
    """
    set_seed(seed)

    # ---------- 1. 读取 testing 数据（2024-01~03） ----------
    cols = [
        "date", "time", "lastPx", "size", "volume",
        "SP1", "BP1", "SV1", "BV1",
        "SP2", "BP2", "SV2", "BV2",
        "SP3", "BP3", "SV3", "BV3",
        "SP4", "BP4", "SV4", "BV4",
        "SP5", "BP5", "SV5", "BV5"
    ]

    tradingDays = ["2023-10-02","2023-10-03","2023-10-04","2023-10-05","2023-10-06","2023-10-11","2023-10-12","2023-10-13","2023-10-16","2023-10-17","2023-10-18","2023-10-19","2023-10-20","2023-10-23","2023-10-24","2023-10-25","2023-10-26","2023-10-27","2023-10-30","2023-10-31","2023-11-01","2023-11-02","2023-11-03","2023-11-06","2023-11-07","2023-11-08","2023-11-09","2023-11-10","2023-11-13","2023-11-14","2023-11-15","2023-11-16","2023-11-17","2023-11-20","2023-11-21","2023-11-22","2023-11-23","2023-11-24","2023-11-27","2023-11-28","2023-11-29","2023-11-30","2023-12-01","2023-12-04","2023-12-05","2023-12-06","2023-12-07","2023-12-08","2023-12-11","2023-12-12","2023-12-13","2023-12-14","2023-12-15","2023-12-18","2023-12-19","2023-12-20","2023-12-21","2023-12-22","2023-12-25","2023-12-26","2023-12-27","2023-12-28","2023-12-29","2024-01-02","2024-01-03","2024-01-04","2024-01-05","2024-01-08","2024-01-09","2024-01-10","2024-01-11","2024-01-12","2024-01-15","2024-01-16","2024-01-17","2024-01-18","2024-01-19","2024-01-22","2024-01-23","2024-01-24","2024-01-25","2024-01-26","2024-01-29","2024-01-30","2024-01-31","2024-02-01","2024-02-02","2024-02-15","2024-02-16","2024-02-19","2024-02-20","2024-02-21","2024-02-22","2024-02-23","2024-02-26","2024-02-27","2024-02-29","2024-03-01","2024-03-04","2024-03-05","2024-03-06","2024-03-07","2024-03-08","2024-03-11","2024-03-12","2024-03-13","2024-03-14","2024-03-15","2024-03-18","2024-03-19","2024-03-20","2024-03-21","2024-03-22","2024-03-25","2024-03-26","2024-03-27","2024-03-28","2024-03-29","2024-04-01","2024-04-02","2024-04-03","2024-04-08","2024-04-09","2024-04-10","2024-04-11","2024-04-12","2024-04-15","2024-04-16","2024-04-17","2024-04-18","2024-04-19","2024-04-22","2024-04-23","2024-04-24","2024-04-25","2024-04-26","2024-04-29","2024-04-30","2024-05-02","2024-05-03","2024-05-06","2024-05-07","2024-05-08","2024-05-09","2024-05-10","2024-05-13","2024-05-14","2024-05-15","2024-05-16","2024-05-17","2024-05-20","2024-05-21","2024-05-22","2024-05-23","2024-05-24","2024-05-27","2024-05-28","2024-05-29","2024-05-30","2024-05-31","2024-06-03","2024-06-04","2024-06-05","2024-06-06","2024-06-07","2024-06-11","2024-06-12","2024-06-13","2024-06-14","2024-06-17","2024-06-18","2024-06-19","2024-06-20","2024-06-21","2024-06-24","2024-06-25","2024-06-26","2024-06-27","2024-06-28","2024-07-01","2024-07-02","2024-07-03","2024-07-04","2024-07-05","2024-07-08","2024-07-09","2024-07-10","2024-07-11","2024-07-12","2024-07-15","2024-07-16","2024-07-17","2024-07-18","2024-07-19","2024-07-22","2024-07-23","2024-07-26","2024-07-29","2024-07-30","2024-07-31","2024-08-01","2024-08-02","2024-08-05","2024-08-06","2024-08-07","2024-08-08","2024-08-09","2024-08-12","2024-08-13","2024-08-14","2024-08-15","2024-08-16","2024-08-19","2024-08-20","2024-08-21","2024-08-22","2024-08-23","2024-08-26","2024-08-27","2024-08-28","2024-08-29","2024-08-30","2024-09-02","2024-09-03","2024-09-04","2024-09-05","2024-09-06","2024-09-09","2024-09-10","2024-09-11","2024-09-12","2024-09-13","2024-09-16","2024-09-18","2024-09-19","2024-09-20","2024-09-23","2024-09-24","2024-09-25","2024-09-26","2024-09-27","2024-09-30","2024-10-01","2024-10-02","2024-10-03","2024-10-04","2024-10-07","2024-10-08","2024-10-09","2024-10-11","2024-10-14","2024-10-15","2024-10-16","2024-10-17","2024-10-18","2024-10-21","2024-10-22","2024-10-23","2024-10-24","2024-10-25","2024-10-28","2024-10-29","2024-10-30","2024-10-31","2024-11-01","2024-11-04","2024-11-05","2024-11-06","2024-11-07","2024-11-08","2024-11-11","2024-11-12","2024-11-13","2024-11-14","2024-11-15","2024-11-18","2024-11-19","2024-11-20","2024-11-21","2024-11-22","2024-11-25","2024-11-26","2024-11-27","2024-11-28","2024-11-29","2024-12-02","2024-12-03","2024-12-04","2024-12-05","2024-12-06","2024-12-09","2024-12-10","2024-12-11","2024-12-12","2024-12-13","2024-12-16","2024-12-17","2024-12-18","2024-12-19","2024-12-20","2024-12-23","2024-12-24","2024-12-25","2024-12-26","2024-12-27","2024-12-30","2024-12-31"]


    print("Raw data loading and processing", stock_code)

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
        print("No raw data to process; exit.")
        return None, None

    # ---------- 2. 转分钟 & 只保留 265 分钟的完整交易日 ----------
    minutelyData = prepareMinutelyData(df, tradingDays)
    print("Minutely data generated.")

    projdata = []
    for _, day_df in minutelyData.groupby('date'):
        if day_df.shape[0] == 265:
            projdata.append(day_df.values)

    projdata = np.array(projdata)   # shape: (N_days, 265, 25)

    if projdata.size == 0:
        print("No full 265-min days found.")
        return None, None

    # ---------- 3. 归一化（和训练完全一致） ----------
    X = projdata[:, :, 5:].astype(float)  # 丢掉前 5 列 meta，只保留 features

    X[:, :, -10:] = np.log(1 + X[:, :, -10:])
    X_mean = X.mean(axis=1)
    X_std = X.std(axis=1)

    X = np.transpose((np.transpose(X, (1, 0, 2)) - X_mean) / (2 * X_std), (1, 0, 2))
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    dataset = MyDataset(torch.tensor(X, dtype=torch.float32))
    test_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    # ---------- 4. 加载模型 ----------
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

    # ---------- 5. 用 Discriminator 对每日序列打分 ----------
    disc_scores = []
    discriminator.eval()
    with torch.no_grad():
        for data in test_dataloader:
            score = discriminator(data)  # shape [batch_size, 1]
            disc_scores.extend(score.squeeze(1).cpu().numpy().tolist())
    disc_scores = np.array(disc_scores)
    assert disc_scores.shape[0] == projdata.shape[0]

    abnormal_mask = disc_scores <= threshold  # True: abnormal day -- 原来的threshold！！！
    # # =======================================
    # #  Quantile-based thresholding  ← NEW
    # #  e.g. choose lowest q fraction as abnormal
    # # =======================================
    # if 0 < threshold < 1:
    #     # interpret threshold as quantile
    #     q_value = np.quantile(disc_scores, threshold)
    #     print(f"Using quantile threshold: q={threshold}, cutoff={q_value:.4f}")
    #     abnormal_mask = disc_scores <= q_value
    # else:
    #     # fallback to raw threshold
    #     abnormal_mask = disc_scores <= threshold

    print(f"Total days: {len(abnormal_mask)}, abnormal days: {abnormal_mask.sum()} "
          f"({abnormal_mask.sum()/len(abnormal_mask):.2%})")

    # ---------- 6. 计算 microstructure 变量（分钟级 snapshot） ----------
    # projdata shape: (N_days, T=265, 25)
    td = projdata  # alias

    # 列索引按你原 columns 的顺序：
    # 0 date, 1 time, 2 lastPx, 3 size, 4 volume,
    # 5 SP5, 6 SP4, 7 SP3, 8 SP2, 9 SP1,
    # 10 BP1, 11 BP2, 12 BP3, 13 BP4, 14 BP5,
    # 15 SV5, 16 SV4, 17 SV3, 18 SV2, 19 SV1,
    # 20 BV1, 21 BV2, 22 BV3, 23 BV4, 24 BV5

    # 强制转成 float，避免 dtype=object
    last_px = td[:, :, 2].astype(float)
    size    = td[:, :, 3].astype(float)

    sp1 = td[:, :, 9].astype(float)
    bp1 = td[:, :, 10].astype(float)

    # SV / BV 也都转 float
    sv1 = td[:, :, 19].astype(float)
    bv1 = td[:, :, 20].astype(float)

    sv5 = td[:, :, 15].astype(float)
    sv4 = td[:, :, 16].astype(float)
    sv3 = td[:, :, 17].astype(float)
    sv2 = td[:, :, 18].astype(float)
    # sv1 已经有了

    bv2 = td[:, :, 21].astype(float)
    bv3 = td[:, :, 22].astype(float)
    bv4 = td[:, :, 23].astype(float)
    bv5 = td[:, :, 24].astype(float)

    # mid quote
    mid = (bp1 + sp1) / 2.0

    # 1. Trade price returns (简单收益)
    trade_ret = last_px[:, 1:] / last_px[:, :-1] - 1.0  # (N_days, T-1)

    # 2. Mid-quote returns
    mid_ret = mid[:, 1:] / mid[:, :-1] - 1.0

    # 3. Trade size
    trade_size = size  # (N_days, T)

    # 4. Bid-ask spread & first diff
    spread = sp1 - bp1                        # (N_days, T)
    spread_diff = spread[:, 1:] - spread[:, :-1]  # (N_days, T-1)

    # 5. 1-level order book pressure
    denom1 = bv1 + sv1
    pressure_1 = (bv1 - sv1) / denom1
    pressure_1[denom1 == 0] = np.nan

    # 6. 5-level order book pressure
    sv_sum = sv5 + sv4 + sv3 + sv2 + sv1
    bv_sum = bv1 + bv2 + bv3 + bv4 + bv5
    denom5 = bv_sum + sv_sum
    pressure_5 = (bv_sum - sv_sum) / denom5
    pressure_5[denom5 == 0] = np.nan


    # ---------- 7. 把 abnormal / normal 的 snapshot 分开 ----------
    # 注意：trade_ret / mid_ret / spread_diff 是 T-1 长度，其他是 T。
    # 但 KS 检验不要求长度一样，只要两组样本都 > 0 即可。

    var_dict = {}

    # flatten 并按 abnormal_mask 选择
    def split_var(arr, is_diff=False):
        """
        arr: shape (N_days, T) or (N_days, T-1)
        is_diff: True 时说明长度 T-1，用同样的 mask 切，不会有问题
        返回: abnormal 1D, normal 1D
        """
        if arr.ndim != 2:
            raise ValueError("arr must be 2D (N_days, T)")
        abn = arr[abnormal_mask, :].ravel()
        nor = arr[~abnormal_mask, :].ravel()
        return abn, nor

    var_dict["trade_return"] = split_var(trade_ret, is_diff=True)
    var_dict["mid_return"] = split_var(mid_ret, is_diff=True)
    var_dict["trade_size"] = split_var(trade_size, is_diff=False)
    var_dict["spread"] = split_var(spread, is_diff=False)
    var_dict["spread_diff"] = split_var(spread_diff, is_diff=True)
    var_dict["pressure_1"] = split_var(pressure_1, is_diff=False)
    var_dict["pressure_5"] = split_var(pressure_5, is_diff=False)

    # ---------- 8. 统计 + KS 检验 ----------
    stats_rows = []
    ks_rows = []

    for var_name, (abn_arr, nor_arr) in var_dict.items():
        abn_clean = _clean_array(abn_arr)
        nor_clean = _clean_array(nor_arr)

        stats_abn = _describe_array(abn_clean)
        stats_nor = _describe_array(nor_clean)

        # 记录描述性统计
        stats_rows.append({
            "stock": stock_code,
            "version": version,
            "variable": var_name,
            "set": "abnormal",
            **stats_abn,
        })
        stats_rows.append({
            "stock": stock_code,
            "version": version,
            "variable": var_name,
            "set": "normal",
            **stats_nor,
        })

        # 做 KS 检验（样本数要>0）
        if len(abn_clean) > 0 and len(nor_clean) > 0:
            ks_stat, ks_p = ks_2samp(abn_clean, nor_clean)
        else:
            ks_stat, ks_p = np.nan, np.nan

        ks_rows.append({
            "stock": stock_code,
            "version": version,
            "variable": var_name,
            "ks_stat": float(ks_stat) if np.isfinite(ks_stat) else np.nan,
            "p_value": float(ks_p) if np.isfinite(ks_p) else np.nan,
            "n_abnormal": int(len(abn_clean)),
            "n_normal": int(len(nor_clean)),
        })

    stats_df = pd.DataFrame(stats_rows)
    ks_df = pd.DataFrame(ks_rows)

    # ---------- 9. 可选：画一个简单 KDE 图看 trade_return 的区别 ----------
    if save_kde_fig:
        abn_tr = _clean_array(var_dict["trade_return"][0])
        nor_tr = _clean_array(var_dict["trade_return"][1])
        plt.figure(figsize=(8, 5))
        if len(abn_tr) > 0:
            sns.kdeplot(abn_tr, label="abnormal trade_return", linestyle="--", fill=True)
        if len(nor_tr) > 0:
            sns.kdeplot(nor_tr, label="normal trade_return")
        plt.title(f"{stock_code} {version} - trade_return KDE (abnormal vs normal)")
        plt.legend()
        fig_path = os.path.join(RESULTS_DIR, f"{stock_code}_trade_return_KDE_{version}.png")
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print("Saved KDE figure to:", fig_path)

    return stats_df, ks_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", type=str, required=True)
    parser.add_argument("--version", type=str, default="raw", choices=["raw", "adjusted"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=307)
    parser.add_argument("--save_kde_fig", action="store_true")
    args = parser.parse_args()

    stats, ks = analyze_microstructure(
        stock_code=args.stock,
        version=args.version,
        threshold=args.threshold,
        batch_size=args.batch_size,
        seed=args.seed,
        save_kde_fig=args.save_kde_fig,
    )
    print(stats.head())
    print(ks.head())
