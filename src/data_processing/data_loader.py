import os
import numpy as np
import pandas as pd


def prepare_minutely_data(df: pd.DataFrame, trading_days: list) -> pd.DataFrame:
    """
    把 tick 级 LOB 数据整理成 1 分钟 snapshot，并只保留交易日 & 09:00~13:25 区间。
    逻辑基本照你原来的 prepareMinutelyData 写法。
    """
    if df.empty:
        return None

    # ----- 1. 基础清洗 & 单位换算 -----
    df["bfValue"] = df["lastPx"] * df["size"]
    df["bfValue"] = df["bfValue"].ffill()
    df["cumValue"] = df.groupby("date")["bfValue"].cumsum()

    df = df[df["SP1"] > 0]
    df = df[df["BP1"] > 0]
    df = df[df["SP1"] - df["BP1"] > 0]

    for i in range(1, 6):
        df[f"SP{i}"] = df[f"SP{i}"] / 100
        df[f"BP{i}"] = df[f"BP{i}"] / 100
        df[f"SV{i}"] = df[f"SV{i}"] * 1000
        df[f"BV{i}"] = df[f"BV{i}"] * 1000

    df["lastPx"] = df["lastPx"] / 100
    df["size"] = df["size"] * 1000
    df["volume"] = df["volume"] * 1000

    df["lastPx"] = df.groupby("date")["lastPx"].ffill()
    df["size"] = df.groupby("date")["size"].transform(lambda x: x.fillna(0))

    df["value"] = df.groupby("date")["cumValue"].diff()
    df["value"] = df["value"].fillna(df["bfValue"])

    df.drop(columns=["bfValue", "cumValue", "value"], inplace=True)

    # ----- 2. 生成时间索引 & 去重 -----
    df_dt = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        format="%Y-%m-%d %H%M%S%f",
    )
    df["dt_index"] = df_dt
    df = df[~df["dt_index"].duplicated(keep="last")]

    # ----- 3. 按分钟聚合（取每分钟最后一个 snapshot） -----
    bin_size = "1min"
    grouped = df.groupby(
        pd.Grouper(key="dt_index", freq=bin_size, closed="right", label="right")
    )
    df_minutely = grouped.last()

    # 明确对每个 SP/BP/SV/BV 再做一次 last（等价但更清晰）
    for i in range(1, 6):
        df_minutely.loc[:, f"SP{i}"] = grouped[f"SP{i}"].last()
        df_minutely.loc[:, f"BP{i}"] = grouped[f"BP{i}"].last()
        df_minutely.loc[:, f"SV{i}"] = grouped[f"SV{i}"].last()
        df_minutely.loc[:, f"BV{i}"] = grouped[f"BV{i}"].last()

    # ----- 4. 只保留交易时段 & 交易日 -----
    df_minutely = df_minutely.between_time("09:00:00", "13:25:00", inclusive="right")
    df_minutely["date"] = df_minutely.index.date
    df_minutely["ttime"] = df_minutely.index.time
    df_minutely.fillna({"time": df_minutely["ttime"]}, inplace=True)
    df_minutely.drop(columns=["ttime"], inplace=True)

    df_minutely = df_minutely[
        df_minutely["date"].astype(str).isin(trading_days)
    ]

    return df_minutely


def data_processing(
    stock: str,
    data_dir: str,
    trading_days: list,
    minutes_per_day: int = 265,
    months=("202401", "202402", "202403"),
):
    """
    读入某只股票在若干月份的 md 数据 (stock_md_YYYYMM_YYYYMM.csv.gz)，
    做：
      1. tick → minutely (prepare_minutely_data)
      2. 过滤掉不完整交易日，只保留 265 分钟的天
      3. 组装 projdata: shape = (N_days, 265, 25)
      4. 做和训练时一致的归一化：得到 X: (N_days, 265, 20)

    返回字典：
      {
        "minutely_df": 所有分钟数据（含所有天）,
        "projdata":    原始 25 维序列 (N_days, 265, 25),
        "X":           归一化后 20 维特征 (N_days, 265, 20),
        "dates":       每个样本对应的交易日数组 (N_days,),
        "columns":     projdata 的列名顺序
      }
    """
    # ---- 1. 读取 gzip CSV ----
    cols = [
        "date", "time", "lastPx", "size", "volume",
        "SP1", "BP1", "SV1", "BV1",
        "SP2", "BP2", "SV2", "BV2",
        "SP3", "BP3", "SV3", "BV3",
        "SP4", "BP4", "SV4", "BV4",
        "SP5", "BP5", "SV5", "BV5",
    ]

    df = pd.DataFrame()
    for m in months:
        file_path = os.path.join(data_dir, f"{stock}_md_{m}_{m}.csv.gz")
        if os.path.exists(file_path):
            df_part = pd.read_csv(file_path, compression="gzip", usecols=cols)
            df = pd.concat([df, df_part], ignore_index=True)
            print(f"Loaded {file_path}")
        else:
            print(f"Skipping missing file {file_path}")

    if df.empty:
        print("No raw data loaded, return None.")
        return None

    # ---- 2. tick → minutely ----
    minutely_df = prepare_minutely_data(df, trading_days)
    print("Minutely data generated.")

    # ---- 3. 组 daily 序列 projdata (N_days, 265, 25) ----
    # 注意列顺序与你脚本保持一致
    columns = [
        "date", "time", "lastPx", "size", "volume",
        "SP5", "SP4", "SP3", "SP2", "SP1",
        "BP1", "BP2", "BP3", "BP4", "BP5",
        "SV5", "SV4", "SV3", "SV2", "SV1",
        "BV1", "BV2", "BV3", "BV4", "BV5",
    ]

    projdata = []
    day_list = []

    for date, day_df in minutely_df.groupby("date"):
        if day_df.shape[0] == minutes_per_day:
            # 按指定列顺序取值，保证 shape 一致
            projdata.append(day_df[columns].values)
            day_list.append(date)

    projdata = np.array(projdata)
    day_list = np.array(day_list)

    print(f"Total days with full {minutes_per_day} minutes: {projdata.shape[0]}")

    if projdata.size == 0:
        print("No full-length trading days after filtering.")
        return None

    # ---- 4. 归一化（和你脚本完全一致）----
    # 丢掉前 5 个 meta 列（date, time, lastPx, size, volume 中只留 feature 部分）
    X = projdata[:, :, 5:].astype(float)  # shape: (N_days, 265, 20)

    # 对最后 10 个量类变量做 log(1+x)
    X[:, :, -10:] = np.log(1 + X[:, :, -10:])

    # 按“日”为单位做标准化
    X_mean = X.mean(axis=1)  # (N_days, 20)
    X_std = X.std(axis=1)    # (N_days, 20)

    # 先把时间轴移到前面便于广播，再移回来
    X = np.transpose((np.transpose(X, (1, 0, 2)) - X_mean) / (2 * X_std), (1, 0, 2))

    # 把 NaN / inf 替换成 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "minutely_df": minutely_df,
        "projdata": projdata,
        "X": X,
        "dates": day_list,
        "columns": columns,
    }
