"""
印度各邦日用电量（面板数据）· 通用模块：数据清洗 + 特征工程

数据集：data/long_data_.csv
    16,599 行，33 个邦/直辖区 × 约 503 天，2020-12-05 为最后一天。
    列：States / Regions / latitude / longitude / Dates / Usage
    一行 = 某个邦某一天的用电量（Usage，量级 1.6 ~ 470）。

这份数据是**面板**（panel）而不是单条序列，和前面三个项目最大的区别
------------------------------------------------------------------
1. 同一天有 33 行（33 个邦），所以任何"按行"的操作（shift / rolling / 切分）
   都必须先按邦分组；否则 shift(1) 会拿到**上一个邦**的值。
2. 日期不连续：有整段缺失（比如 2019-01-19 之后直接跳到 02-10）。
   所以要把「邦 × 完整日期」补成网格，否则 shift(7) 不是"上周同一天"。
3. 有重复记录：同一个 (邦, 日期) 出现两次且数值不同（第 6140 / 6173 行）。
   按 (邦, 日期) 去重，并统计"值冲突"的条数。
4. 切分要**按日期**切，不能按行切：否则同一天的不同邦会分到训练和测试两边。

其它约定（和前三个项目一致）
----------------------------
- 常量集中在顶部；
- 列名统一 snake_case；
- 滞后/滚动只看过去，滚动先 shift(1) 再 rolling；
- 训练和预测共用 build_features；
- 特征清单缺列直接报错。
"""

import numpy as np
import pandas as pd

# ---------------- 全局常量 ----------------
FREQ = '1D'
TARGET = 'usage'

# 滞后：按"天"给，代码里会转成同一邦内的 shift 步数
LAG_STEPS = {'1d': 1, '2d': 2, '7d': 7, '14d': 14}
# 滚动：7 天 / 14 天 / 30 天
ROLL_STEPS = {'7d': 7, '14d': 14, '30d': 30}

# 目标缺失最多向前补几天（2 天能盖住一个周末）
TARGET_FILL_LIMIT = 2

# 静态列：每个邦固定不变，补网格后会变 NaN，需要按邦回填
STATIC_COLS = ['region', 'lat', 'lon']

# 重命名
RENAME = {
    'States': 'state',
    'Regions': 'region',
    'latitude': 'lat',
    'longitude': 'lon',
    'Dates': 'date',
    'Usage': 'usage',
}

# =========================================================
# 1. 数据清洗
# =========================================================
def data_preprocessing(path, logger=None):
    """读 CSV -> 统一列名 -> 解析日期 -> 去重 -> 补「邦 × 日期」网格 -> 缺失处理。

    :return: DataFrame，列包含 state / date / usage / region / lat / lon
             以及一列 usage_is_filled（1 表示这一行的 usage 是补出来的，不参与训练）
    """
    def log(msg):
        if logger: logger.info(msg)
        else: print(msg)

    # 读取与校验列名
    df = pd.read_csv(path, encoding='utf-8-sig')
    missing = [c for c in RENAME if c not in df.columns]
    if missing: raise KeyError(f'CSV 缺少列：{missing}\n实际列名：{list(df.columns)}')

    # 解析日期
    # 把数据转成 datetime 格式，而且印度的日期格式是 DD/MM/YYYY，故而dayfirst=True
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    # 统计解析失败的行数，isna()是检查 NaN 的，sum()是求和
    n_bad = int(df['date'].isna().sum())
    # 删除解析失败的行，保留有效行
    df = df.dropna(subset=['date'])
    # 统计删除的行数
    n_drop = len(df) - n_bad

    log(f'[data] 日期解析失败 {n_bad} 行（dayfirst=True，DD/MM/YYYY）')
    log(f'[data] 日期范围 {df["date"].min().date()} ~ {df["date"].max().date()}，'
        f'共 {df["date"].nunique()} 个不同日期')
    log(f'[data] 删除 {n_drop} 行无效日期')

    # 3) 类型转换
    # 把目标列和经纬度列转成数值型，errors='coerce' 是把无法转换的值转成 NaN
    for c in [TARGET, 'lat', 'lon']:
        # 转成数值型，无法转换的值转成 NaN，to_numeric 是 pandas 的函数,用来转换为数值类型
        df[c] = pd.to_numeric(df[c], errors='coerce')


    # 4) 按 (邦, 日期) 去重：重复记录里数值可能冲突.
    # 去重，keep=False 保留所有重复行
    dup = df.duplicated(subset=['state', 'date'], keep=False)
    # 统计重复行数
    n_dup_rows = int(dup.sum())
    # 统计重复行中值冲突的行数
    if n_dup_rows:
        # 按 (邦, 日期) 分组，统计目标列的唯一值个数
        conflict = df[dup].groupby(['state', 'date'])[TARGET].nunique()
        # 统计值冲突的行数
        n_conflict = int((conflict > 1).sum())
        log(f'[data] 重复记录 {n_dup_rows} 行，涉及 {len(conflict)} 个 (邦,日期)，'
            f'其中 {n_conflict} 个的值不相同 → 保留第一条')

    # 去重，保留第一条
    df = df.drop_duplicates(subset=['state', 'date'], keep='first')

    # 0) 确保 (state, date) 唯一
    df = df.drop_duplicates(subset=['state', 'date'], keep='first')

    # 1) 邦与完整日期网格
    states = sorted(df['state'].unique().tolist())
    dates = pd.date_range(df['date'].min(), df['date'].max(), freq=FREQ)

    # 校验原始日期是否都在网格里，防止 reindex 丢行
    missing = set(pd.to_datetime(df['date'].unique())) - set(dates)
    if missing:
        log(f'[warn] {len(missing)} 个原始日期不在 FREQ={FREQ} 网格中，可能被丢弃')

    full_idx = pd.MultiIndex.from_product([states, dates], names=['state', 'date'])

    # 2) reindex 补网格
    n_before = len(df)
    df = df.set_index(['powerplant', 'date']).reindex(full_idx).reset_index()
    n_add = len(df) - n_before
    log(f'[data] 补网格：{n_before} 行 -> {len(df)} 行'
        f'（{len(states)} 邦 × {len(dates)} 个 {FREQ} 周期，补 {n_add} 行空记录）')

    # 6) 静态列回填（region / lat / lon 每个邦是常数，补网格后变 NaN）
    for c in STATIC_COLS:
        df[c] = df.groupby('state')[c].ffill().bfill()

    # 7) 目标缺失：短缺口补历史，但标记"补出来的行"
    n_missing = int(df[TARGET].isna().sum())
    df['usage_is_filled'] = df[TARGET].isna().astype('int8')
    df[TARGET] = df[TARGET].ffill(limit=TARGET_FILL_LIMIT)
    n_left = int(df[TARGET].isna().sum())
    log(f'[data] 目标缺失 {n_missing} 行（{n_missing / len(df):.1%}），'
        f'其中 {n_missing - n_left} 行用前值补齐（不参与训练），仍有 {n_left} 行空缺')

    # 8) 覆盖率：每个邦实际有多少天的数据
    cover = df.groupby('state')[TARGET].apply(lambda s: float(s.notna().mean()))
    cover = cover.sort_values()
    log('[data] 覆盖率最低的 5 个邦：'
        + '，'.join(f'{k}={v:.1%}' for k, v in cover.head(5).items()))
    log(f'[data] 整体覆盖率 {df[TARGET].notna().mean():.1%}')

    q = df[TARGET].quantile([0, 0.01, 0.5, 0.99, 1.0])
    log(f'[data] 用电量分位：min={q.iloc[0]:.1f}, p1={q.iloc[1]:.1f}, '
        f'p50={q.iloc[2]:.1f}, p99={q.iloc[3]:.1f}, max={q.iloc[4]:.1f}')

    return df.sort_values(['state', 'date']).reset_index(drop=True)

def mape(y_true, y_pred):
    """平均绝对百分比误差（%）。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.clip(np.abs(y_true), 1e-6, None)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)

# =========================================================
# 2. 特征工程
# =========================================================
def _add_onehot(df, col, categories, prefix, logger=None):
    """固定类别 one-hot：即使某个类别在数据里没出现，也会生成全 0 列。"""
    if col not in df.columns:
        raise KeyError(f'[onehot] 列 {col!r} 不存在')

    cat = pd.Categorical(df[col], categories=list(categories))

    mask_unknown = df[col].notna() & pd.isna(cat)
    n_unknown = int(mask_unknown.sum())
    if n_unknown and logger is not None:
        sample = pd.unique(df.loc[mask_unknown, col])[:10].tolist()
        logger.warning(f'[onehot] {col} 有 {n_unknown} 个值不在固定类别里，'
                       f'已按全 0 处理，样例={sample}')

    dummies = pd.get_dummies(cat, prefix=prefix, dtype='int8')
    dummies.index = df.index

    clash = [c for c in dummies.columns if c in df.columns]
    if clash:
        raise ValueError(f'[onehot] {prefix} 生成的列名与已有列冲突: {clash[:5]}')

    result = pd.concat([df, dummies], axis=1)
    return result, list(dummies.columns)


def build_features(df, logger=None):
    """统一的特征工程：训练与预测必须都调用它。

    注意：这是面板数据，所有 lag / rolling 都必须**按邦分组**，
    否则 shift(1) 会取到上一行（可能是另一个邦）。

    :param df: data_preprocessing 的输出
    :return: (含特征的 DataFrame, 特征列名 list)
    """

    def log(msg):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    result = df.copy()
    result['date'] = pd.to_datetime(result['date'])
    # 面板的"时间顺序"是 (邦, 日期)，排序必须两个键一起排
    result = result.sort_values(['state', 'date']).reset_index(drop=True)

    # ---- 1) 日历特征 ----
    result['dow'] = result['date'].dt.dayofweek  # 0=周一
    result['month'] = result['date'].dt.month
    result['day'] = result['date'].dt.day
    result['dayofyear'] = result['date'].dt.dayofyear
    result['is_weekend'] = (result['dow'] >= 5).astype('int8')

    # 1) 时间特征的 sin / cos 转换，周期分别为 7 天和 365 天
    result['dow_sin'] = np.sin(2 * np.pi * result['dow'] / 7)
    result['dow_cos'] = np.cos(2 * np.pi * result['dow'] / 7)
    result['doy_sin'] = np.sin(2 * np.pi * result['dayofyear'] / 365)
    result['doy_cos'] = np.cos(2 * np.pi * result['dayofyear'] / 365)

    # ---- 2) 邦的 one-hot ----
    # 邦名从数据里取（33 个），并打印出来；以后新增邦要重新训练模型
    state_cats = sorted(result['state'].dropna().unique().tolist())
    # 添加 one-hot 列
    result, state_onehot = _add_onehot(result, 'state', state_cats, 'st', logger)
    log(f'[feature] 邦 one-hot：{len(state_onehot)} 个')

    # ---- 3) 地理坐标 ----
    # 添加地理坐标列
    geo_cols = ['lat', 'lon']

    # ---- 4) 滞后 + 滚动：全部按邦分组，全部只向过去看 ----
    grouped = result.groupby('state', sort=False)[TARGET]

    # 添加滞后列
    lag_cols = []
    for name, step in LAG_STEPS.items():
        col = f'lag_{name}'
        result[col] = grouped.shift(step)
        lag_cols.append(col)

    # 添加滚动列
    roll_cols = []
    for name, step in ROLL_STEPS.items():
        min_p = max(2, step // 2)
        mean_col, std_col = f'roll_{name}_mean', f'roll_{name}_std'
        result[mean_col] = grouped.transform(
            lambda s, w=step, m=min_p: s.shift(1).rolling(w, min_periods=m).mean())
        result[std_col] = grouped.transform(
            lambda s, w=step, m=min_p: s.shift(1).rolling(w, min_periods=m).std())
        roll_cols += [mean_col, std_col]

    # ---- 5) 特征清单 ----
    cal_cols = ['dow', 'month', 'day', 'dayofyear', 'is_weekend',
                'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos']
    features = (state_onehot + cal_cols + geo_cols + lag_cols + roll_cols)

    lost = [c for c in features if c not in result.columns]
    if lost:
        raise KeyError(f'以下特征列不存在，检查列名拼写：{lost}')
    features = [c for c in features if c in result.columns]

    if len(features) != len(set(features)):
        raise ValueError('特征列名重复，检查 one-hot 前缀')

    log(f'[feature] 输出 {len(result)} 行 × {len(features)} 个特征')
    log(f'[feature] 特征清单：{features}')
    return result, features


