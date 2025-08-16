#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
动态熵权法 + GM(1,1) 趋势绩效综合评价模型实现

依据 `problem/思路分析.md` 中描述的建模思路：
1. 读取已标准化后的数据集（列值范围已在 [0,1]，方向已统一为“越大越好”）。
2. 按季度分别计算各指标熵值与熵权（季度内离散度越大权重越高）。
3. 计算季度综合绩效得分 s_{it} = Σ w_{jt} * x_{ijt}。
4. 对每个人的季度得分序列应用 GM(1,1) 模型估计发展系数 α_i 与灰作用量 β_i，提取趋势值 trend_i。
5. 计算平均季度绩效 \bar{s}_i，并组合最终得分 S_i = \bar{s}_i + alpha * trend_i （可选对 trend 做归一化）。
6. 输出：
   - 每季度指标熵值/权重表
   - 人员季度得分表
   - GM(1,1) 参数与趋势表
   - 最终综合排名、前五(不排序集合)与前三(排序)

使用说明：
python dynamic_entropy_gm_model.py --input ../build/标准化数据集.csv --output ../build --alpha 0.3

参数：
--alpha           趋势项权重 α ∈ [0,1]
--normalize-trend 是否对趋势值进行 Min-Max 归一化后再融合（默认开启，避免量纲不匹配）
--top-unordered   不排序奖励人数 (默认 5)
--top-ranked      排序奖励人数 (默认 3)

依赖：pandas, numpy
"""

from __future__ import annotations

import os
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict

# 可选绘图依赖（若未安装 matplotlib 则跳过可视化）
try:
    import matplotlib
    matplotlib.use('Agg')  # 后端设为无界面
    import matplotlib.pyplot as plt
    # 配置中文字体，避免输出图中中文变成方框。按候选列表找到系统已安装字体。
    try:
        from matplotlib import font_manager
        candidate_fonts = [
            'SimHei',            # 黑体
            'Microsoft YaHei',   # 微软雅黑
            'Microsoft YaHei UI',
            'STHeiti',
            'Heiti TC',
            'PingFang SC',
            'Source Han Sans CN',
            'Source Han Sans SC',
            'Arial Unicode MS'
        ]
        available = set(f.name for f in font_manager.fontManager.ttflist)
        chosen = None
        for f in candidate_fonts:
            if f in available:
                chosen = f
                break
        if chosen:
            matplotlib.rcParams['font.sans-serif'] = [chosen]
            matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        else:
            # 若无中文字体，退回默认但依然避免负号问题
            matplotlib.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass
except Exception:  # noqa
    plt = None

EPS = 1e-12

@dataclass
class GMResult:
    person: str
    alpha_hat: float
    beta_hat: float
    trend_raw: float
    trend: float  # 可能归一化后的趋势
    points_used: int
    predicted_next: float | None = None  # 预测下一期原始得分（季度得分）

def compute_entropy_weights(df: pd.DataFrame, indicator_cols: List[str]) -> pd.DataFrame:
    """按给定数据子集（单季度）计算熵值与权重。"""
    data = df[indicator_cols].copy().astype(float).values
    data = np.where(data <= 0, EPS, data)
    col_sum = data.sum(axis=0)
    col_sum = np.where(col_sum == 0, EPS, col_sum)
    p = data / col_sum
    m = data.shape[0]
    entropy = - (p * np.log(p)).sum(axis=0) / np.log(m)
    diversity = 1 - entropy
    if np.isclose(diversity.sum(), 0):
        weights = np.ones_like(diversity) / len(diversity)
    else:
        weights = diversity / diversity.sum()
    return pd.DataFrame({
        '指标': indicator_cols,
        '熵值': entropy,
        '差异性系数': diversity,
        '权重': weights
    })

def gm11_trend(series: List[float]) -> Dict[str, float]:
    """GM(1,1) 参数估计、趋势值与下一期预测。

    返回:
        alpha_hat, beta_hat, trend_raw, predicted_next
        其中 predicted_next 为预测下一期原始序列值 x0_hat(n+1)。
    """
    x0 = np.array(series, dtype=float)
    n = len(x0)
    if n < 2 or np.allclose(x0.std(), 0):
        mean_val = float(np.mean(x0)) if n > 0 else 0.0
        return dict(alpha_hat=0.0, beta_hat=mean_val, trend_raw=mean_val, predicted_next=mean_val)
    x1 = np.cumsum(x0)
    z = 0.5 * (x1[1:] + x1[:-1])
    B = np.column_stack((-z, np.ones(n - 1)))
    Y = x0[1:]
    try:
        params = np.linalg.inv(B.T @ B) @ B.T @ Y
        a_hat, b_hat = params[0], params[1]
    except np.linalg.LinAlgError:
        mean_val = float(np.mean(x0))
        return dict(alpha_hat=0.0, beta_hat=mean_val, trend_raw=mean_val, predicted_next=mean_val)
    if abs(a_hat) < 1e-8:
        trend_raw = float(b_hat)
        # 线性近似: 下一期预测用最后一期值
        predicted_next = float(x0[-1])
    else:
        trend_raw = float((b_hat / a_hat) * (1 - math.exp(-a_hat)))
        # GM(1,1) 解：x1_hat(k) = (x0(1) - b/a)*exp(-a*(k-1)) + b/a
        # 原序列预测: x0_hat(k) = x1_hat(k) - x1_hat(k-1)
        c = x0[0] - b_hat / a_hat
        x1_hat_prev = (c * math.exp(-a_hat * (n - 1))) + b_hat / a_hat
        x1_hat_next = (c * math.exp(-a_hat * n)) + b_hat / a_hat
        predicted_next = float(x1_hat_next - x1_hat_prev)
    return dict(alpha_hat=float(a_hat), beta_hat=float(b_hat), trend_raw=trend_raw, predicted_next=predicted_next)

def rename_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    empty_count = 0
    for c in df.columns:
        if (c is None) or (str(c).strip() == ''):
            empty_count += 1
            new_cols.append(f'空列_{empty_count}')
        else:
            new_cols.append(str(c).strip())
    df.columns = new_cols
    return df

def main():
    # ===== 可配置区域 =====
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    INPUT_PATH = os.path.join(PROJECT_ROOT, 'build','problem1', '标准化数据集.csv')  # 标准化数据集路径
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'build','problem1')                     # 输出目录
    ALPHA = 0.3              # 趋势项权重 α ∈ [0,1] （若 AUTO_ALPHA=True 将被覆盖）
    NORMALIZE_TREND = True   # 是否对趋势值做 Min-Max 归一化
    TOP_UNORDERED_N = 5      # 不排序奖励人数
    TOP_RANKED_N = 3         # 排序奖励人数
    AUTO_ALPHA = True        # 是否自动寻优 alpha
    ALPHA_GRID = np.arange(0.0, 1, 0.05)  # 寻优网格 (含0)
    PLOT_FIGSIZE = (8, 5)    # 可视化图尺寸 (宽, 高) 英寸
    PLOT_DPI = 240           # 图像 DPI 提升分辨率
    # 目标：使综合得分与“预测下一期得分”具有最高 Spearman 相关，同时保留一定区分度。
    # 复合目标函数 J = corr_spearman(S, predicted_next) + 0.2 * standardized(range)
    # ======================

    alpha = max(0.0, min(1.0, ALPHA))
    normalize_trend = NORMALIZE_TREND
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_PATH, encoding='utf-8-sig')
    df = rename_empty_columns(df)

    base_cols = {'人员', '季度', '单位', '季度得分'}
    indicator_cols = [c for c in df.columns if c not in base_cols]
    if not indicator_cols:
        raise ValueError('未发现指标列，请检查输入文件。')

    quarter_scores_records = []
    weights_all_quarters = []
    quarters = sorted(df['季度'].unique())

    for q in quarters:
        q_df = df[df['季度'] == q].reset_index(drop=True)
        weights_df = compute_entropy_weights(q_df, indicator_cols)
        weights_df.insert(0, '季度', q)
        weights_all_quarters.append(weights_df)
        weight_map = dict(zip(weights_df['指标'], weights_df['权重']))
        scores = []
        for _, row in q_df.iterrows():
            s = 0.0
            for col in indicator_cols:
                val = row[col]
                if pd.notna(val):
                    s += float(val) * weight_map[col]
            scores.append(s)
        q_df = q_df.copy()
        q_df['动态熵权季度得分'] = scores
        for _, row in q_df[['人员', '单位', '动态熵权季度得分']].iterrows():
            quarter_scores_records.append({'人员': row['人员'], '季度': q, '单位': row['单位'], '季度得分': row['动态熵权季度得分']})

    quarter_scores_df = pd.DataFrame(quarter_scores_records)

    gm_results: List[GMResult] = []
    for person, group in quarter_scores_df.groupby('人员'):
        series = []
        for q in quarters:
            rows = group[group['季度'] == q]
            if not rows.empty:
                series.append(float(rows['季度得分'].iloc[0]))
        gm_dict = gm11_trend(series)
        gm_results.append(GMResult(person=person,
                                   alpha_hat=gm_dict['alpha_hat'],
                                   beta_hat=gm_dict['beta_hat'],
                                   trend_raw=gm_dict['trend_raw'],
                                   trend=gm_dict['trend_raw'],
                                   points_used=len(series),
                                   predicted_next=gm_dict.get('predicted_next')))

    if normalize_trend and gm_results:
        raw_vals = np.array([g.trend_raw for g in gm_results], dtype=float)
        min_v, max_v = raw_vals.min(), raw_vals.max()
        if not np.isclose(max_v, min_v):
            norm = (raw_vals - min_v) / (max_v - min_v)
        else:
            norm = np.ones_like(raw_vals) * 0.5
        for g, nv in zip(gm_results, norm):
            g.trend = float(nv)

    gm_df = pd.DataFrame([{'人员': g.person,
                           'GM_alpha_hat': g.alpha_hat,
                           'GM_beta_hat': g.beta_hat,
                           '趋势_raw': g.trend_raw,
                           '趋势值': g.trend,
                           '预测下一期得分': g.predicted_next,
                           '序列点数': g.points_used} for g in gm_results])

    avg_scores = quarter_scores_df.groupby('人员')['季度得分'].mean().reset_index().rename(columns={'季度得分': '平均季度绩效'})
    merged_base = avg_scores.merge(gm_df[['人员', '趋势值', '预测下一期得分']], on='人员', how='left')
    merged_base['趋势值'].fillna(0, inplace=True)

    alpha_used = alpha
    alpha_search_table = None
    if AUTO_ALPHA:
        # === 方案1: 使用标准化后的 avg 与 trend 进行 alpha 网格寻优 (不改变最终得分计算) ===
        rows_metrics = []
        pred_next = merged_base['预测下一期得分']
        if pred_next.isna().any():
            pred_next = pred_next.fillna(pred_next.mean())
        avg_col = merged_base['平均季度绩效'].astype(float)
        trend_col = merged_base['趋势值'].astype(float)
        avg_mean, avg_std = float(avg_col.mean()), float(avg_col.std(ddof=0))
        trend_mean, trend_std = float(trend_col.mean()), float(trend_col.std(ddof=0))
        if avg_std < 1e-12:
            avg_z = np.zeros_like(avg_col, dtype=float)
        else:
            avg_z = (avg_col - avg_mean) / avg_std
        if trend_std < 1e-12:
            trend_z = np.zeros_like(trend_col, dtype=float)
        else:
            trend_z = (trend_col - trend_mean) / trend_std

        for a in ALPHA_GRID:
            S_z = avg_z + a * trend_z  # 仅用于择优的标准化组合
            rng = S_z.max() - S_z.min()
            std = S_z.std(ddof=0)
            if S_z.nunique() > 1 and pred_next.nunique() > 1:
                corr = S_z.rank(method='average').corr(pred_next.rank(method='average'))
            else:
                corr = 0.0
            rows_metrics.append({'alpha': a, 'corr_pred_next': corr, 'range': rng, 'std': std})
        alpha_df = pd.DataFrame(rows_metrics)
        # 归一化 range
        if alpha_df['range'].max() - alpha_df['range'].min() > 0:
            alpha_df['range_norm'] = (alpha_df['range'] - alpha_df['range'].min()) / (alpha_df['range'].max() - alpha_df['range'].min())
        else:
            alpha_df['range_norm'] = 0.0
        # 复合目标 J
        alpha_df['J'] = alpha_df['corr_pred_next'] + 0.2 * alpha_df['range_norm']
        # 选择 J 最大的 alpha；若并列取较小 alpha（保守）
        best_row = alpha_df.sort_values(['J', 'alpha'], ascending=[False, True]).iloc[0]
        alpha_used = float(best_row['alpha'])
        alpha_search_table = alpha_df
        # === 可视化 alpha-J 曲线 ===
        if plt is not None and not alpha_df.empty:
            try:
                fig, ax1 = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
                ax1.plot(alpha_df['alpha'], alpha_df['J'], marker='o', label='J(复合目标)')
                ax1.plot(alpha_df['alpha'], alpha_df['corr_pred_next'], marker='s', linestyle='--', label='Spearman相关')
                ax1.plot(alpha_df['alpha'], alpha_df['range_norm'], marker='^', linestyle=':', label='range_norm')
                ax1.axvline(alpha_used, color='red', linestyle='-.', label=f'最佳α={alpha_used:.2f}')
                ax1.set_xlabel('alpha')
                ax1.set_ylabel('指标值 (标准化 J / 相关 / range_norm)')
                ax1.set_title('Alpha 敏感性与复合目标 J')
                ax1.grid(alpha=0.3, linestyle=':')
                ax1.legend(fontsize=8)
                fig.tight_layout()
                alpha_plot_path = os.path.join(OUTPUT_DIR, 'alpha_sensitivity.png')
                fig.savefig(alpha_plot_path, dpi=PLOT_DPI, bbox_inches='tight')
                plt.close(fig)
            except Exception as _e:  # 捕获绘图失败不影响主流程
                alpha_plot_path = None
        else:
            alpha_plot_path = None
    else:
        alpha_plot_path = None
    # 用最终 alpha_used 计算得分
    merged = merged_base.copy()
    merged['最终综合得分'] = merged['平均季度绩效'] + alpha_used * merged['趋势值']
    merged = merged.sort_values('最终综合得分', ascending=False).reset_index(drop=True)

    top_unordered_n = max(1, TOP_UNORDERED_N)
    top_ranked_n = max(1, TOP_RANKED_N)
    top_ranked_df = merged.head(top_ranked_n).copy()
    top_unordered_df = merged.head(top_unordered_n).copy()
    unordered_set = set(top_unordered_df['人员'].tolist())
    threshold = float(top_unordered_df['最终综合得分'].min()) if not top_unordered_df.empty else 0.0

    weights_concat = pd.concat(weights_all_quarters, ignore_index=True)
    excel_path = os.path.join(OUTPUT_DIR, '动态熵权_GM模型结果.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        weights_concat.to_excel(writer, sheet_name='季度指标权重', index=False)
        quarter_scores_df.to_excel(writer, sheet_name='季度得分', index=False)
        gm_df.to_excel(writer, sheet_name='GM趋势参数', index=False)
        merged.to_excel(writer, sheet_name='最终综合排名', index=False)
        top_ranked_df.to_excel(writer, sheet_name=f'前{top_ranked_n}_排序', index=False)
        top_unordered_df.to_excel(writer, sheet_name=f'前{top_unordered_n}_不排序', index=False)
        if alpha_search_table is not None:
            alpha_search_table.to_excel(writer, sheet_name='Alpha敏感性', index=False)

    summary = {
        'alpha_input': alpha,
        'alpha_used': alpha_used,
        'auto_alpha': AUTO_ALPHA,
        'alpha_grid': list(ALPHA_GRID if AUTO_ALPHA else [alpha]),
        'normalize_trend': normalize_trend,
    'alpha_selection_standardized': True if AUTO_ALPHA else False,
        'threshold': threshold,
        'top_unordered_n': top_unordered_n,
        'top_ranked_n': top_ranked_n,
        'top_unordered_persons': list(unordered_set),
    'top_ranked': top_ranked_df[['人员', '最终综合得分']].to_dict('records'),
    'alpha_plot': 'alpha_sensitivity.png' if (AUTO_ALPHA and alpha_plot_path) else None
    }
    pd.Series(summary).to_json(os.path.join(OUTPUT_DIR, 'dynamic_model_summary.json'), force_ascii=False, indent=2)

    print("=== 动态熵权 + GM(1,1) 模型完成 ===")
    print(f"输入文件: {INPUT_PATH}")
    print(f"季度数量: {len(quarters)} | 指标数量: {len(indicator_cols)} | 人员数: {len(merged)}")
    print(f"趋势权重 alpha_used = {alpha_used} (输入默认 {alpha}) | AUTO_ALPHA={AUTO_ALPHA} | 趋势归一化: {normalize_trend}")
    print(f"不排序奖励人数: {top_unordered_n} (阈值={threshold:.6f}) -> 人员集合: {unordered_set}")
    print(f"排序奖励前{top_ranked_n}:")
    for i, r in top_ranked_df.iterrows():
        print(f"  第{i+1}名: {r['人员']} - 最终综合得分={r['最终综合得分']:.6f} (平均季度绩效={r['平均季度绩效']:.6f}, 趋势值={r['趋势值']:.6f})")
    print(f"结果已保存: {excel_path}")

if __name__ == '__main__':
    main()
