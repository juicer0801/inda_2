# -*- coding: utf-8 -*-
"""
印度各邦日用电量（面板数据）· 预测 / 回测脚本

设计要点
--------
1. 特征由 utils.common.build_features 生成，和训练是同一份代码；
2. 列名、列序从 bundle['features'] 取，缺列报错、多列忽略；
3. 面板数据：一次预测所有邦、所有选中的日期；
4. 评估时剔除"标签是补出来的"行（usage_is_filled == 1）；
5. 默认回测最后 30 天，改 START_DATE / END_DATE 可换区间。

注意：这是"一步预测"口径（每个邦当天用它自己之前的历史）。
      如果要提前一周预测，把 lag_1d/lag_2d 从特征里去掉、只保留 lag_7d 及更早的滞后。

运行：python src/predict.py
"""
import os
import sys
import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------- 项目路径 ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.log import Logger
from utils.common import TARGET, mape, data_preprocessing, build_features

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
FIG_DIR = os.path.join(DATA_DIR, 'fig')
OUT_DIR = os.path.join(DATA_DIR, 'output')
for _d in (FIG_DIR, OUT_DIR):
    os.makedirs(_d, exist_ok=True)

INPUT_FILE = os.path.join(DATA_DIR, 'long_data_.csv')
BUNDLE_PATH = os.path.join(MODEL_DIR, 'xgb_bundle.pkl')

# None = 默认最后 30 天；也可以写成 '2020-09-01' / '2020-09-01 00:00:00'
START_DATE = None
END_DATE = None

# 画哪个邦的曲线
STATE_TO_PLOT = 'Delhi'

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12


def predict_best(model, X):
    """只用 best_iteration 对应的树预测（不同版本行为一致）。"""
    best = getattr(model, 'best_iteration', None)
    if best is None:
        return model.predict(X)
    try:
        return model.predict(X, iteration_range=(0, int(best) + 1))
    except TypeError:
        return model.predict(X, ntree_limit=int(best) + 1)


def main():
    ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    logger = Logger(PROJECT_ROOT, 'predict_' + ts).get_logger()
    logger.info('=' * 60)

    # 1) 加载 bundle
    if not os.path.exists(BUNDLE_PATH):
        raise FileNotFoundError(f'找不到 {BUNDLE_PATH}，请先运行 src/train.py')

    bundle = joblib.load(BUNDLE_PATH)
    if not isinstance(bundle, dict) or 'features' not in bundle:
        raise TypeError('模型文件不是 bundle 格式，请用当前 train.py 重新训练')

    model = bundle['model']
    features = list(bundle['features'])
    meta = bundle.get('meta', {})
    logger.info('加载模型：%d 个特征 / %s 个邦，训练于 %s，测试 MAE=%s',
                len(features), meta.get('n_states'), meta.get('train_time'),
                meta.get('metrics', [{}])[0].get('MAE') if meta.get('metrics') else 'N/A')

    # 2) 和训练完全相同的清洗 + 特征
    raw = data_preprocessing(INPUT_FILE, logger)
    data, built_features = build_features(raw, logger)

    # 3) 列对齐
    missing = [c for c in features if c not in data.columns]
    if missing:
        raise KeyError(f'缺少训练用到的特征（前 10 个）：{missing[:10]}\n'
                       f'说明数据和训练时不一致，请确认 train.py 是同一版本')
    extra = [c for c in built_features if c not in features]
    if extra:
        logger.info('当前多出 %d 列特征（已忽略）：%s', len(extra), extra)

    # 4) 选日期区间
    end = pd.Timestamp(END_DATE) if END_DATE else data['date'].max()
    start = pd.Timestamp(START_DATE) if START_DATE else end - pd.Timedelta(days=30)

    target = data[(data['date'] >= start) & (data['date'] <= end)].copy()
    need_cols = [c for c in features if c.startswith(('lag_', 'roll_'))]
    target = target.dropna(subset=need_cols)      # 不 reset_index，保持和 data 的索引对齐

    if target.empty:
        raise ValueError(f'{start} ~ {end} 没有可预测的行。\n'
                         f'数据实际范围 {data["date"].min().date()} ~ {data["date"].max().date()}')

    logger.info('预测区间：%s ~ %s，共 %d 天 × %d 个邦 = %d 行',
                target['date'].min().date(), target['date'].max().date(),
                target['date'].nunique(), target['state'].nunique(), len(target))

    # 5) 预测（整表一次算完）
    X = target.reindex(columns=features)
    y_hat = predict_best(model, X)

    result = pd.DataFrame({
        '日期': target['date'].values,
        '邦': target['state'].values,
        '真实值': target[TARGET].values,
        '预测值': y_hat,
        '标签是否补齐': target['usage_is_filled'].values,
    })
    result['误差'] = result['真实值'] - result['预测值']

    # 6) 评估：只算"标签不是补出来的"行
    valid = result['真实值'].notna().to_numpy() & (result['标签是否补齐'].to_numpy() == 0)
    if valid.any():
        yt = result['真实值'].to_numpy()[valid]
        yp = result['预测值'].to_numpy()[valid]

        mae = float(mean_absolute_error(yt, yp))
        logger.info('有效样本 %d / %d', int(valid.sum()), len(result))
        logger.info('MAE=%.2f  RMSE=%.2f  MAPE=%.2f%%  bias=%.2f',
                    mae, float(np.sqrt(mean_squared_error(yt, yp))),
                    mape(yt, yp), float(np.mean(yt - yp)))

        if 'lag_1d' in target.columns:
            base = target['lag_1d'].to_numpy()[valid]
            ok = ~np.isnan(base)
            if ok.any():
                base_mae = float(mean_absolute_error(yt[ok], base[ok]))
                logger.info('基线-昨天 MAE=%.2f，模型提升 %.2f%%',
                            base_mae, (base_mae - mae) / base_mae * 100)

        # 各邦误差
        tmp = result[valid].copy()
        tmp['abs_err'] = (tmp['真实值'] - tmp['预测值']).abs()
        by_state = tmp.groupby('邦')['abs_err'].agg(['mean', 'size'])
        by_state.columns = ['MAE', '样本数']
        logger.info('各邦 MAE（最差 8 个）:\n%s',
                    by_state.sort_values('MAE', ascending=False).head(8).round(2).to_string())

        # 7) 画一个邦的曲线
        sub = result[result['邦'] == STATE_TO_PLOT].sort_values('日期')
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(16, 5))
            ax.plot(sub['日期'], sub['真实值'], label='真实值', marker='o', markersize=3)
            ax.plot(sub['日期'], sub['预测值'], label='预测值', marker='x', markersize=3)
            ax.set_title(f'{STATE_TO_PLOT}：{sub["日期"].min().date()} ~ {sub["日期"].max().date()}')
            ax.set_ylabel('Usage')
            ax.legend()
            fig.tight_layout()
            fig_path = os.path.join(FIG_DIR, f'预测效果_{ts}.png')
            fig.savefig(fig_path, dpi=100)
            plt.close(fig)
            logger.info('预测图已保存：%s', fig_path)

    csv_path = os.path.join(OUT_DIR, f'pred_{ts}.csv')
    result.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info('预测结果已保存：%s', csv_path)
    logger.info('=' * 60)


if __name__ == '__main__':
    main()
