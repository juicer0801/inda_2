"""
印度各邦日用电量（面板数据）· 训练脚本

流程：清洗 -> 特征 -> **按日期**切分 -> 5 折（按日期）体检 -> 训练 -> 评估 -> 保存 bundle

和前面三个项目最大的不同：这是面板数据（33 个邦 × 503 天），所以
--------------------------------------------------------------------------
1. 切分必须按**日期**切，不能按行切：
   按行切会把"同一天的不同邦"分到训练和测试两边，等于偷看答案。
2. 交叉验证也必须按日期做折，不能用 TimeSeriesSplit(行)：
   因为行顺序是"邦1的全部日期 → 邦2的全部日期…"，
   按行切会让某一折只包含某几个邦。
3. 所有 lag / rolling 都在 common.build_features 里按邦分组算好了。

运行：python src/train.py
"""
import io
import json
import os
import sys
import datetime
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ---------- 项目路径 ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.log import Logger
from utils.common import (FREQ, LAG_STEPS, ROLL_STEPS, TARGET,
                          TARGET_FILL_LIMIT, mape, data_preprocessing, build_features)

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
FIG_DIR = os.path.join(DATA_DIR, 'fig')
for _d in (MODEL_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)

INPUT_FILE = os.path.join(DATA_DIR, 'long_data_.csv')

# ---------- 切分与训练参数（按"日期"切） ----------
TEST_RATIO = 0.20      # 最后 20% 的日期做测试
VAL_RATIO = 0.15       # 训练日期里再取最后 15% 做早停
SEED = 42

PARAMS = dict(
    n_estimators=1500,
    learning_rate=0.03,
    max_depth=7,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    gamma=0.1,
    objective='reg:squarederror',
    eval_metric='mae',
    random_state=SEED,
    n_jobs=-1,
)
EARLY_STOPPING_ROUNDS = 50

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12


# =========================================================
# 基础工具
# =========================================================
def fit_xgb(X_tr, y_tr, X_val, y_val, params, logger=None):
    """带早停的 XGBoost，兼容 early_stopping_rounds 的两种位置。"""
    params = {k: v for k, v in params.items() if k != 'early_stopping_rounds'}

    model = XGBRegressor(**params)
    try:
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False,
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    except TypeError as err:
        if 'early_stopping_rounds' not in str(err):
            raise
        model = XGBRegressor(early_stopping_rounds=EARLY_STOPPING_ROUNDS, **params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    if logger is not None:
        logger.info('best_iteration=%s', getattr(model, 'best_iteration', 'N/A'))
    return model


def predict_best(model, X):
    """只用 best_iteration 对应的树预测。"""
    best = getattr(model, 'best_iteration', None)
    if best is None:
        return model.predict(X)
    try:
        return model.predict(X, iteration_range=(0, int(best) + 1))
    except TypeError:
        return model.predict(X, ntree_limit=int(best) + 1)


def evaluate(y_true, y_pred, name=''):
    """一次算齐指标。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    return {
        '集合': name,
        '样本数': len(y_true),
        'MAE': mae,
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAPE%': mape(y_true, y_pred),
        'R2': float(r2_score(y_true, y_pred)),
        'bias': float(np.mean(y_true - y_pred)),
        'MAE/均值%': mae / float(np.mean(y_true)) * 100,
    }


def split_by_date(df, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO):
    """按日期切三份，保证同一天的所有邦进同一个集合。

    这是面板数据最容易出错的地方：按行切（iloc[:n]）会把同一天的不同邦拆开。
    """
    dates = pd.DatetimeIndex(sorted(df['date'].unique()))
    test_start = dates[int(len(dates) * (1 - test_ratio))]

    train_df = df[df['date'] < test_start].copy()
    test_df = df[df['date'] >= test_start].copy()

    train_dates = pd.DatetimeIndex(sorted(train_df['date'].unique()))
    val_start = train_dates[int(len(train_dates) * (1 - val_ratio))]

    tr = train_df[train_df['date'] < val_start].copy()
    va = train_df[train_df['date'] >= val_start].copy()
    return tr, va, test_df


def cross_validate_by_date(df, features, params, n_splits=5):
    """按日期做扩展窗口交叉验证（面板数据不能用 TimeSeriesSplit(行)）。

    第 k 折：训练 = 前 k 段日期，验证 = 第 k+1 段日期；
    再从训练段内部取最后 10% 的日期做早停。
    """
    dates = pd.DatetimeIndex(sorted(df['date'].unique()))
    fold = len(dates) // (n_splits + 1)
    if fold < 5:
        return pd.DataFrame()

    rows = []
    for k in range(1, n_splits + 1):
        tr = df[df['date'] < dates[fold * k]]
        te = df[(df['date'] >= dates[fold * k]) & (df['date'] < dates[fold * (k + 1)])]
        if len(tr) == 0 or len(te) == 0:
            continue

        inner_dates = pd.DatetimeIndex(sorted(tr['date'].unique()))
        inner_cut = inner_dates[int(len(inner_dates) * 0.9)]
        itr = tr[tr['date'] < inner_cut]
        iva = tr[tr['date'] >= inner_cut]
        if len(iva) == 0:
            continue

        model = fit_xgb(itr[features], itr[TARGET], iva[features], iva[TARGET], params)
        pred = predict_best(model, te[features])

        rows.append({
            'fold': k,
            '训练日期数': len(inner_dates),
            'n_train': len(itr),
            'MAE': float(mean_absolute_error(te[TARGET], pred)),
            'RMSE': float(np.sqrt(mean_squared_error(te[TARGET], pred))),
            'best_iter': int(getattr(model, 'best_iteration', -1)),
        })
    return pd.DataFrame(rows)


# =========================================================
# 数据分布可视化
# =========================================================
def view_data_distribution(data, logger):
    d = data.copy()
    d['month_period'] = d['date'].dt.to_period('M')
    d['dow'] = d['date'].dt.dayofweek

    buffer = io.StringIO()
    d.info(buf=buffer)
    logger.info('数据整体信息:\n%s', buffer.getvalue())
    logger.info('数据前5行:\n%s', d.head().to_string())

    fig, axes = plt.subplots(3, 2, figsize=(16, 15))

    # ① 用电量分布（注意各邦量级差很多，1.6 ~ 470）
    axes[0, 0].hist(d[TARGET].dropna(), bins=100)
    axes[0, 0].set_title('用电量分布（33 个邦混在一起）')
    axes[0, 0].set_xlabel('Usage')

    # ② 月度总用电量：COVID 封锁（2020-03~05）会在这里露出一大块缺口
    monthly = d.groupby('month_period')[TARGET].sum()
    axes[0, 1].plot(monthly.index.astype(str), monthly.values, marker='o', color='b')
    axes[0, 1].set_title('月度总用电量（所有邦合计）')
    axes[0, 1].tick_params(axis='x', rotation=90)

    # ③ 星期效应
    dow_avg = d.groupby('dow')[TARGET].mean()
    axes[1, 0].bar(dow_avg.index, dow_avg.values, color='g')
    axes[1, 0].set_title('各星期平均用电量（0=周一）')
    axes[1, 0].set_xticks(range(7))
    axes[1, 0].set_xticklabels(['一', '二', '三', '四', '五', '六', '日'])

    # ④ 各邦平均用电量 Top15（量级差一个数量级，模型必须知道"是哪个邦"）
    state_avg = d.groupby('state')[TARGET].mean().sort_values(ascending=False).head(15)[::-1]
    axes[1, 1].barh(state_avg.index, state_avg.values, color='steelblue')
    axes[1, 1].set_title('各邦平均用电量 Top15')
    axes[1, 1].set_xlabel('Usage')

    # ⑤ 覆盖率：每个邦实际有多少天有数据
    cover = d.groupby('state')[TARGET].apply(lambda s: float(s.notna().mean()) * 100)
    cover = cover.sort_values().head(15)
    axes[2, 0].barh(cover.index, cover.values, color='crimson')
    axes[2, 0].set_title('覆盖率最低的 15 个邦（%）')
    axes[2, 0].set_xlabel('%')

    # ⑥ 三条代表邦的时序，看量级和形态差异
    for s in ['Delhi', 'Maharashtra', 'UP']:
        sub = d[d['state'] == s].sort_values('date')
        if not sub.empty:
            axes[2, 1].plot(sub['date'], sub[TARGET], label=s, linewidth=0.9)
    axes[2, 1].set_title('三个代表邦的用电量时序')
    axes[2, 1].legend(fontsize=9)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, '数据分布.png')
    fig.savefig(path, dpi=100)
    plt.close(fig)
    logger.info('数据分布图已保存：%s', path)


# =========================================================
# 绘图
# =========================================================
def plot_state_pred(err, state, path, title, days=30):
    sub = err[err['state'] == state].sort_values('time')
    if sub.empty:
        return
    start = sub['time'].min()
    sub = sub[sub['time'] < start + pd.Timedelta(days=days)]
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(sub['time'], sub['y'], label='真实值', marker='o', markersize=3)
    ax.plot(sub['time'], sub['pred'], label='预测值', marker='x', markersize=3)
    ax.set_title(title)
    ax.set_ylabel('Usage')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_importance(importance, path, top=20):
    sub = importance.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(sub))))
    ax.barh(sub.index.astype(str), sub.values, color='steelblue')
    ax.set_title(f'特征重要性 Top{len(sub)}')
    ax.set_xlabel('importance')
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# =========================================================
# 主流程
# =========================================================
def main():
    ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    logger = Logger(PROJECT_ROOT, 'train_' + ts).get_logger()
    t0 = time.time()

    logger.info('=' * 70)
    logger.info('各邦用电量预测（面板）：xgboost %s / pandas %s',
                xgboost.__version__, pd.__version__)

    # 1) 数据
    raw = data_preprocessing(INPUT_FILE, logger)
    view_data_distribution(raw, logger)

    # 2) 特征（lag / rolling 已在 common 里按邦分组算好）
    data, features = build_features(raw, logger)

    # 3) 两条过滤：先去掉"标签是补出来的"行，再丢掉 lag 窗口凑不齐的行
    n0 = len(data)
    data = data[data['usage_is_filled'] == 0].copy()
    logger.info('去掉"标签是补出来的"行：%d -> %d（-%d）', n0, len(data), n0 - len(data))

    need_cols = ([TARGET]
                 + [f'lag_{k}' for k in LAG_STEPS]
                 + [f'roll_{k}_{s}' for k in ROLL_STEPS for s in ('mean', 'std')])
    n1 = len(data)
    data = data.dropna(subset=need_cols).reset_index(drop=True)
    logger.info('dropna（滞后窗口）：%d -> %d，丢弃 %d 行（含每邦前 14 天预热期）',
                n1, len(data), n1 - len(data))
    if len(data) < 1000:
        raise ValueError('可用样本不足，检查数据或把 lag 窗口调小')

    # 4) 按日期切分（不是按行！）
    train_df, val_df, test_df = split_by_date(data)
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError('切分后有空集合，检查日期范围或 TEST_RATIO')

    logger.info('训练 %d 行 / %d 天（%s ~ %s）', len(train_df),
                train_df['date'].nunique(), train_df['date'].min().date(), train_df['date'].max().date())
    logger.info('验证 %d 行 / %d 天（%s ~ %s）', len(val_df),
                val_df['date'].nunique(), val_df['date'].min().date(), val_df['date'].max().date())
    logger.info('测试 %d 行 / %d 天（%s ~ %s）', len(test_df),
                test_df['date'].nunique(), test_df['date'].min().date(), test_df['date'].max().date())

    X_tr, y_tr = train_df[features], train_df[TARGET]
    X_val, y_val = val_df[features], val_df[TARGET]
    X_te, y_te = test_df[features], test_df[TARGET]

    # 5) 按日期做 5 折体检
    logger.info('开始按日期 5 折 ...')
    cv = cross_validate_by_date(data, features, PARAMS)
    if not cv.empty:
        logger.info('CV 结果:\n%s', cv.round(2).to_string(index=False))
        logger.info('CV MAE: %.1f ± %.1f', cv['MAE'].mean(), cv['MAE'].std())

    # 6) 正式训练
    logger.info('开始正式训练 ...')
    model = fit_xgb(X_tr, y_tr, X_val, y_val, PARAMS, logger)
    y_pred = predict_best(model, X_te)

    # 7) 指标 + 基线
    xgb_row = evaluate(y_te, y_pred, name='XGB')
    base_d1_row = evaluate(y_te, test_df['lag_1d'], name='基线-昨天')
    base_d7_row = evaluate(y_te, test_df['lag_7d'], name='基线-上周同一天')

    xgb_row['较昨天提升%'] = (base_d1_row['MAE'] - xgb_row['MAE']) / base_d1_row['MAE'] * 100
    xgb_row['较上周提升%'] = (base_d7_row['MAE'] - xgb_row['MAE']) / base_d7_row['MAE'] * 100

    metrics_df = pd.DataFrame([xgb_row, base_d1_row, base_d7_row])
    logger.info('测试集指标:\n%s', metrics_df.round(2).to_string(index=False))

    # 8) 误差归因
    err = pd.DataFrame({
        'time': test_df['date'].values,
        'state': test_df['state'].values,
        'y': y_te.values,
        'pred': y_pred,
    })
    err['error'] = err['y'] - err['pred']          # 正 = 低估
    err['abs_error'] = err['error'].abs()
    err['month'] = err['time'].dt.month

    month_tbl = err.groupby('month').agg(
        n=('y', 'size'), MAE=('abs_error', 'mean'), bias=('error', 'mean'),
        true_mean=('y', 'mean'), pred_mean=('pred', 'mean'),
    )
    logger.info('分月表现:\n%s', month_tbl.reset_index().round(2).to_string(index=False))

    state_tbl = err.groupby('state').agg(
        n=('y', 'size'), MAE=('abs_error', 'mean'), bias=('error', 'mean'),
        true_mean=('y', 'mean'),
    )
    state_tbl['MAE%'] = state_tbl['MAE'] / state_tbl['true_mean'] * 100
    logger.info('误差最大的 10 个邦:\n%s',
                state_tbl.sort_values('MAE%', ascending=False).head(10).round(2).to_string())

    daily = err.groupby(err['time'].dt.date)['abs_error'].mean().sort_values(ascending=False)
    logger.info('误差最大的 10 天:\n%s', daily.head(10).round(1).to_string())

    # 9) 图
    worst_state = state_tbl['MAE%'].idxmax()
    plot_state_pred(err, worst_state,
                    os.path.join(FIG_DIR, f'最差邦_{ts}.png'),
                    f'{worst_state}：测试期前 30 天 真实 vs 预测')
    plot_state_pred(err, 'Delhi',
                    os.path.join(FIG_DIR, f'Delhi_{ts}.png'),
                    'Delhi：测试期前 30 天 真实 vs 预测')

    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    logger.info('Top20 特征重要性:\n%s', importance.head(20).round(4).to_string())
    lag_in = [c for c in importance.index if c.startswith(('lag_', 'roll_'))]
    logger.info('滞后/滚动特征重要性合计 %.4f', importance[lag_in].sum())
    plot_importance(importance, os.path.join(FIG_DIR, f'特征重要性_{ts}.png'))

    # 10) 保存
    meta = {
        'train_time': datetime.datetime.now().isoformat(timespec='seconds'),
        'data_start': str(data['date'].min().date()),
        'data_end': str(data['date'].max().date()),
        'train_end': str(train_df['date'].max().date()),
        'val_start': str(val_df['date'].min().date()),
        'test_start': str(test_df['date'].min().date()),
        'n_rows': int(len(data)),
        'n_features': int(len(features)),
        'n_states': int(data['state'].nunique()),
        'n_train': int(len(train_df)), 'n_val': int(len(val_df)), 'n_test': int(len(test_df)),
        'test_ratio': TEST_RATIO, 'val_ratio': VAL_RATIO,
        'target_fill_limit': TARGET_FILL_LIMIT,
        'best_iteration': int(getattr(model, 'best_iteration', -1)),
        'params': PARAMS,
        'metrics': json.loads(metrics_df.round(4).to_json(orient='records', force_ascii=False)),
        'versions': {'xgboost': xgboost.__version__,
                     'pandas': pd.__version__, 'numpy': np.__version__},
    }

    bundle_path = os.path.join(MODEL_DIR, 'xgb_bundle.pkl')
    joblib.dump({'model': model, 'features': features, 'target': TARGET,
                 'freq': FREQ, 'meta': meta}, bundle_path)
    logger.info('模型已保存：%s', bundle_path)

    with open(os.path.join(MODEL_DIR, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(MODEL_DIR, f'metrics_{ts}.csv'), 'w', encoding='utf-8-sig') as f:
        metrics_df.to_csv(f, index=False)

    logger.info('总耗时 %.1f 秒', time.time() - t0)
    logger.info('=' * 70)


if __name__ == '__main__':
    main()
