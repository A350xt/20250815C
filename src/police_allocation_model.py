import pandas as pd
import numpy as np
import re
from pathlib import Path
from itertools import product
try:
    from sklearn.linear_model import Ridge
except Exception:  # 若未安装则后续回退 OLS
    Ridge = None

"""模型二：警力配置合理性评估 + 新增警力优化分配
使用说明:
1. 先运行 data_preprocessing.py 生成 build/problem2/processed_data_raw.csv
2. 再运行本脚本:
   python src/police_allocation_model.py
可选参数通过修改脚本顶部常量。
输出文件:
  build/problem2/regression_summary.txt 回归参数/标准化信息
  build/problem2/predicted_police.csv    预测与缺编
  build/problem2/allocation_best.csv     最优分配方案
  build/problem2/allocation_top5.csv     目标值最优前5方案
"""

# ---------------- 可调参数 ----------------
W1 = 0.7            # 警力平衡权重
W2 = 0.3            # 绩效提升权重
ALPHA = 0.5         # k_i 中缺编部分权重
BETA = 0.5          # k_i 中绩效不足部分权重 (ALPHA + BETA = 1)
TOTAL_NEW = 7       # 新增警力总数
TOP_K = 5           # 输出前K个方案
SAFETY_FACTOR = 1.08   # 工作负荷法安全冗余系数 (>1 增加需求)
USE_WORKLOAD_METHOD = True  # 是否启用工作负荷容量法作为主预测
# 下面新增若干控制参数（可调）：
TRIM_LOWER = 0.2          # 计算工作负荷/警力 单位容量时的下分位裁剪
TRIM_UPPER = 0.8          # 上分位裁剪 (避免极端值拉低容量导致预测过高)
SHORTAGE_CAP_RATE = 0.15  # 总缺编率上限（相对现有总警力）。若预测缺编>该比例，则自动缩放
ENABLE_TOTAL_CALIBRATION = True  # 是否启用总量校准
AUTO_TUNE_SAFETY = True          # 是否对工作负荷预测再做二分缩放以达到目标缺编率
TARGET_SHORTAGE_RATE = 0.08      # 目标缺编率（相对现有总警力），AUTO_TUNE_SAFETY 为 True 时生效
SAFETY_SCALE_MIN = 0.40          # 二分搜索缩放下界
SAFETY_SCALE_MAX = 1.60          # 二分搜索缩放上界
SAFETY_SCALE_ITERS = 18          # 二分迭代次数（精度 ~ (max-min)/2^iters）
workload_capacity_used = None
workload_scale_used = None
workload_auto_scale_used = None
workload_original_shortage_rate = None
workload_final_shortage_rate = None

BASE = Path(r'C:/Users/cfcsn/OneDrive/本科个人资料/数学建模暑假培训/20250815C')
DATA_DIR = BASE / 'build' / 'problem2'
RAW_FILE = DATA_DIR / 'processed_data_raw.csv'
PERF_FILE = BASE / 'problem' / 'C题：附件1-治安所长业务绩效.xls'  # (旧) 如不再使用可忽略
STANDARD_PERF_FILE = BASE / 'build' / 'problem1' / '标准化数据集.csv'  # 新：从标准化数据集中提取季度得分

OUT_REG_SUMMARY = DATA_DIR / 'regression_summary.txt'
OUT_PRED = DATA_DIR / 'predicted_police.csv'
OUT_ALLOC_BEST = DATA_DIR / 'allocation_best.csv'
OUT_ALLOC_TOP = DATA_DIR / 'allocation_top5.csv'

# ---------------- 读取特征数据 ----------------
if not RAW_FILE.exists():
    raise FileNotFoundError(f'缺少预处理输出文件: {RAW_FILE}')

df = pd.read_csv(RAW_FILE)

# 必要列检查 / 回退
REQUIRED_BASE = ['Index','现有警力数','总工作负荷时长','is_scenic_spot']
for col in REQUIRED_BASE:
    if col not in df.columns:
        raise ValueError(f'缺少必要列 {col}，请先更新预处理脚本')

# 选择自变量 (按思路分析二: 三年平均值)。若 *_年均 列存在则优先; 否则用单列。
# 面积 行业场所 人口(常/暂/寄) 110警情 刑事立案 打处数 治安案件查处 工作负荷 景区哑变量
feature_candidates = []

def pick(col_names):
    for c in col_names:
        if c in df.columns:
            return c
    return None

# 面积与行业场所（可能仅一列）
feature_candidates.append(pick(['区域面积']))
feature_candidates.append(pick(['行业场所数']))

# 人口年均
for root in ['常住人口','暂住人口','寄住人口']:
    feature_candidates.append(pick([f'{root}_年均', root]))

# 警情与案件 (平均) —— 名称兼容
feature_candidates.append(pick(['平均110警情数','110警情数_三年均值','110警情数']))
feature_candidates.append(pick(['平均刑事案件立案数','刑事案件立案数_三年均值','刑事案件立案数']))
feature_candidates.append(pick(['平均打处数','打处数_三年均值','打处数']))
feature_candidates.append(pick(['平均治安案件查处数','治安案件查处数_三年均值','治安案件查处数']))

# 工作负荷 & 景区变量
feature_candidates.append('总工作负荷时长')
feature_candidates.append('is_scenic_spot')

features = [c for c in feature_candidates if c is not None]
features = list(dict.fromkeys(features))  # 去重
print('[DEBUG] 初始候选自变量:', features)

# 强制数值化 + 统计缺失
X_raw = df[features].apply(pd.to_numeric, errors='coerce')
numeric_na_counts = X_raw.isna().sum().to_dict()
print('[DEBUG] 各列缺失数:', numeric_na_counts)

# 若全部 NaN 列，剔除
all_nan_cols = [c for c in X_raw.columns if X_raw[c].isna().all()]
if all_nan_cols:
    print('[WARN] 全部为空列剔除:', all_nan_cols)
    X_raw.drop(columns=all_nan_cols, inplace=True)
    features = [c for c in features if c not in all_nan_cols]

# 缺失以列均值填充（列均值若仍 NaN 则填 0）
for c in X_raw.columns:
    col_mean = X_raw[c].mean()
    if np.isnan(col_mean):
        col_mean = 0.0
    X_raw[c] = X_raw[c].fillna(col_mean)

# 去除常量列
constant_cols = [c for c in X_raw.columns if np.isclose(X_raw[c].std(ddof=0), 0)]
if constant_cols:
    print('[WARN] 常量列剔除:', constant_cols)
    X_raw.drop(columns=constant_cols, inplace=True)
    features = [c for c in features if c not in constant_cols]

if not features:
    raise ValueError('无有效自变量特征，无法回归。')

# 标准化
means = {}
stds = {}
X_std = X_raw.copy()
for col in features:
    if col == 'is_scenic_spot':
        means[col] = 0.0
        stds[col] = 1.0
        # 保持 0/1
        continue
    m = X_raw[col].mean()
    s = X_raw[col].std(ddof=0)
    if s == 0 or np.isnan(s):
        s = 1.0
    means[col] = m
    stds[col] = s
    X_std[col] = (X_raw[col] - m)/s

Y = pd.to_numeric(df['现有警力数'], errors='coerce').astype(float)
if Y.isna().any():
    raise ValueError('现有警力数列存在非数值无法转换。')

X_mat = np.column_stack([np.ones(len(X_std))] + [X_std[c].values for c in features])

# 检查是否有 NaN
row_mask = ~np.isnan(X_mat).any(axis=1) & ~np.isnan(Y)
if row_mask.sum() < len(Y):
    print(f'[WARN] 存在含 NaN 行 {len(Y)-row_mask.sum()} 条，已剔除。')
    X_mat = X_mat[row_mask]
    Y = Y[row_mask]
    df = df.loc[row_mask].reset_index(drop=True)

# 回归（原始全特征 OLS）
XtX = X_mat.T @ X_mat
try:
    beta = np.linalg.solve(XtX, X_mat.T @ Y)
except np.linalg.LinAlgError:
    beta = np.linalg.pinv(XtX) @ X_mat.T @ Y
Y_pred_reg = X_mat @ beta
if np.isnan(Y_pred_reg).any():
    raise RuntimeError('预测结果存在 NaN，请检查特征或数据。')
Y_pred_reg_round = np.ceil(Y_pred_reg).astype(int)  # 使用上取整更敏感
D_reg = np.maximum(0, Y_pred_reg_round - Y)

# ===== 工作负荷容量法 =====
if USE_WORKLOAD_METHOD:
    if '总工作负荷时长' not in df.columns:
        raise ValueError('缺少 总工作负荷时长 列，无法使用工作负荷容量法')
    workload = pd.to_numeric(df['总工作负荷时长'], errors='coerce')
    cap_series = workload / np.where(df['现有警力数']>0, df['现有警力数'], np.nan)
    cap_series = cap_series.replace([np.inf, -np.inf], np.nan)
    # 分位裁剪，去掉极端容量(极小的容量值会导致预测需求暴涨)
    valid_caps = cap_series.dropna().values
    if len(valid_caps) == 0:
        capacity = 1.0
    else:
        qL = np.quantile(valid_caps, TRIM_LOWER)
        qU = np.quantile(valid_caps, TRIM_UPPER)
        trimmed = valid_caps[(valid_caps >= qL) & (valid_caps <= qU)]
        if len(trimmed) == 0:
            trimmed = valid_caps
        capacity = np.median(trimmed)
    if not np.isfinite(capacity) or capacity <= 0:
        capacity = np.nanmedian(cap_series)
    if not np.isfinite(capacity) or capacity <= 0:
        capacity = 1.0
    workload_capacity_used = capacity
    Y_pred_workload = (workload / capacity) * SAFETY_FACTOR
    # 初步预测完成后，若缺编过大则进行总量校准（仅针对工作负荷法基线）
    Y_pred_workload_round = np.ceil(Y_pred_workload).astype(int)
    D_workload = np.maximum(0, Y_pred_workload_round - Y)
    R_workload = np.maximum(0, Y - Y_pred_workload_round)
    # 记录原始缺编率
    workload_original_shortage_rate = D_workload.sum() / Y.sum() if Y.sum()>0 else 0
    if ENABLE_TOTAL_CALIBRATION:
        total_existing = Y.sum()
        total_shortage = D_workload.sum()
        # 若缺编率超过上限 -> 缩放预测
        if total_shortage / total_existing > SHORTAGE_CAP_RATE:
            target_total = total_existing * (1 + SHORTAGE_CAP_RATE)
            sum_pred_current = Y_pred_workload_round.sum()
            # 缩放系数(限制<=1避免放大)
            scale = min(1.0, target_total / sum_pred_current)
            workload_scale_used = scale if scale < 1.0 else 1.0
            if scale < 1.0:
                Y_pred_workload = Y_pred_workload * scale
                Y_pred_workload_round = np.ceil(Y_pred_workload).astype(int)
                D_workload = np.maximum(0, Y_pred_workload_round - Y)
                R_workload = np.maximum(0, Y - Y_pred_workload_round)
                print(f'[INFO] 工作负荷预测总缺编率 {total_shortage/total_existing:.2%} 超过上限, 已按系数 {scale:.3f} 缩放. 新缺编率 {(D_workload.sum()/total_existing):.2%}')
    # 二分自动调参（在总量校准之后执行，使结果更贴近目标缺编率）
    if AUTO_TUNE_SAFETY and Y.sum()>0:
        target = TARGET_SHORTAGE_RATE
        if target < 0: target = 0
        # 若当前缺编率已经接近目标（±0.5%）则不再调整
        current_rate = D_workload.sum()/Y.sum()
        if abs(current_rate - target) > 0.005:
            base_pred = (workload / capacity) * SAFETY_FACTOR  # 未附加额外比例的“基础”
            # 定义函数：给定scale -> 缺编率
            def shortage_rate(scale):
                pred = base_pred * scale
                pred_round = np.ceil(pred).astype(int)
                D_tmp = np.maximum(0, pred_round - Y)
                rate = D_tmp.sum()/Y.sum()
                return rate, pred, pred_round, D_tmp
            lo, hi = SAFETY_SCALE_MIN, SAFETY_SCALE_MAX
            # 保证区间两端覆盖目标（若不覆盖则扩展一次）
            rate_lo, *_ = shortage_rate(lo)
            rate_hi, *_ = shortage_rate(hi)
            expand_cnt = 0
            while (target < rate_lo or target > rate_hi) and expand_cnt < 4:
                # 扩展区间
                lo = max(0.05, lo*0.5)
                hi = hi*1.5
                rate_lo, *_ = shortage_rate(lo)
                rate_hi, *_ = shortage_rate(hi)
                expand_cnt += 1
            best_tuple = (abs(current_rate - target), 1.0, Y_pred_workload, Y_pred_workload_round, D_workload)
            # 若当前已是最优初始
            if abs(current_rate - target) < best_tuple[0]:
                best_tuple = (abs(current_rate - target), 1.0, Y_pred_workload, Y_pred_workload_round, D_workload)
            # 二分
            for _ in range(SAFETY_SCALE_ITERS):
                mid = 0.5*(lo+hi)
                rate_mid, pred_mid, pred_mid_round, D_mid = shortage_rate(mid)
                diff = abs(rate_mid - target)
                if diff < best_tuple[0]:
                    best_tuple = (diff, mid, pred_mid, pred_mid_round, D_mid)
                # 根据单调性调整区间（scale ↑ -> 预测↑ -> 缺编率↑）
                if rate_mid > target:
                    hi = mid
                else:
                    lo = mid
            # 采用最佳
            _, scale_best, pred_best, pred_best_round, D_best = best_tuple
            Y_pred_workload = pred_best
            Y_pred_workload_round = pred_best_round
            D_workload = D_best
            R_workload = np.maximum(0, Y - Y_pred_workload_round)
            workload_auto_scale_used = scale_best
            workload_final_shortage_rate = D_workload.sum()/Y.sum()
            print(f"[AUTO] 缺编率二分调参完成: 原始 {current_rate:.2%} -> 目标 {target:.2%} -> 新 {workload_final_shortage_rate:.2%} (scale={scale_best:.3f})")
        else:
            workload_final_shortage_rate = current_rate
else:
    Y_pred_workload = None
    Y_pred_workload_round = None
    D_workload = None
    R_workload = None

# ===== 精简特征 Ridge 回归（防止过拟合） =====
ridge_features = [c for c in ['总工作负荷时长','平均110警情数','平均刑事案件立案数','平均治安案件查处数','is_scenic_spot'] if c in df.columns]
Y_pred_ridge = None
if Ridge is not None and len(ridge_features) >= 2:
    X_ridge_raw = df[ridge_features].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    # 标准化
    X_ridge = (X_ridge_raw - X_ridge_raw.mean())/X_ridge_raw.std(ddof=0).replace(0,1)
    ridge = Ridge(alpha=5.0, fit_intercept=True, random_state=0)
    ridge.fit(X_ridge, Y)
    Y_pred_ridge = ridge.predict(X_ridge)
    Y_pred_ridge_round = np.ceil(Y_pred_ridge).astype(int)
    D_ridge = np.maximum(0, Y_pred_ridge_round - Y)
else:
    Y_pred_ridge_round = None
    D_ridge = None

# ===== 选择主预测基线 =====
if USE_WORKLOAD_METHOD and D_workload is not None and D_workload.sum() > 0:
    baseline_method = 'workload'
    baseline_pred = Y_pred_workload
    baseline_pred_round = Y_pred_workload_round
    D = D_workload
else:
    # 若工作负荷全 0 则看 ridge
    if D_ridge is not None and D_ridge.sum() > 0:
        baseline_method = 'ridge'
        baseline_pred = Y_pred_ridge
        baseline_pred_round = Y_pred_ridge_round
        D = D_ridge
    else:
        baseline_method = 'reg_ols'
        baseline_pred = Y_pred_reg
        baseline_pred_round = Y_pred_reg_round
        D = D_reg

# 合并结果
pred_df = df[['Index','现有警力数']].copy()
pred_df['预测_OLS'] = Y_pred_reg
pred_df['预测_OLS_ceil'] = Y_pred_reg_round
if Y_pred_workload is not None:
    pred_df['预测_工作负荷'] = Y_pred_workload
    pred_df['预测_工作负荷_ceil'] = Y_pred_workload_round
if Y_pred_ridge is not None:
    pred_df['预测_Ridge'] = Y_pred_ridge
    pred_df['预测_Ridge_ceil'] = Y_pred_ridge_round
pred_df['采用方法'] = baseline_method
pred_df['预测警力数'] = baseline_pred
pred_df['预测警力_取整'] = baseline_pred_round
pred_df['缺编人数D'] = D
pred_df['缺编比例'] = D / np.where(baseline_pred_round==0, 1, baseline_pred_round)
if D_workload is not None:
    pred_df['缺编_工作负荷'] = D_workload
    pred_df['冗余_工作负荷'] = R_workload

"""绩效读取改造：优先从标准化数据集(问题1输出)中提取【季度得分】按派出所聚合。
逻辑:
1. 读取 STANDARD_PERF_FILE (CSV)
2. 识别列: '单位' (含 'A0派出所' 等), '季度得分'
3. 提取派出所代码 A\d+ -> 计算该派出所季度得分均值
4. 归一化 (除以最大值) 得到 0~1 绩效P
5. 若文件不存在或失败 -> 回退为全 1
"""
performance = {}
if STANDARD_PERF_FILE.exists():
    try:
        std_perf = pd.read_csv(STANDARD_PERF_FILE, encoding='utf-8')
        # 兼容空列名: 重命名空字符串 -> '空列'
        std_perf.columns = [c if (isinstance(c,str) and c.strip()!='') else f'空列_{i}' for i,c in enumerate(std_perf.columns)]
        if '单位' in std_perf.columns:
            # 确保 季度得分 列存在; 若不存在尝试尾列
            score_col = '季度得分' if '季度得分' in std_perf.columns else std_perf.columns[-1]
            tmp = std_perf[['单位', score_col]].copy()
            tmp[score_col] = pd.to_numeric(tmp[score_col], errors='coerce')
            tmp['Index'] = tmp['单位'].astype(str).str.extract(r'(A\d+)')
            grp = tmp.dropna(subset=['Index', score_col]).groupby('Index')[score_col].mean()
            if not grp.empty:
                max_score = grp.max()
                if max_score <= 0:
                    max_score = 1.0
                performance = (grp / max_score).to_dict()
                print('[INFO] 已从标准化数据集聚合季度得分为绩效, 覆盖派出所数:', len(performance))
    except Exception as e:
        print('[WARN] 读取标准化绩效数据失败，回退全1:', e)
else:
    print('[WARN] 标准化绩效文件不存在，回退全1:', STANDARD_PERF_FILE)

# 回退: 未覆盖的派出所设为均值或 1.0
if performance:
    avg_val = float(np.mean(list(performance.values())))
    for idx in pred_df['Index']:
        if idx not in performance:
            performance[idx] = avg_val
else:
    for idx in pred_df['Index']:
        performance[idx] = 1.0

pred_df['绩效P'] = pred_df['Index'].map(performance)

# ---------------- 计算 k_i ----------------
pred_df['缺编人数D'] = D  # 确保同步
D_sum = D.sum() if D.sum() > 0 else 1
k_i = ALPHA * (pred_df['缺编人数D'].values / D_sum) + BETA * (1 - pred_df['绩效P'].values)
pred_df['k_i'] = k_i

# ---------------- 枚举分配  ----------------
# 生成所有满足 sum x = TOTAL_NEW 的非负整数向量 (11维) —— 使用递归减少组合量
indices = pred_df['Index'].tolist()

best_solutions = []  # list of (F, x_vector)

# 递归生成
x = [0]*len(indices)

def dfs(pos, remaining):
    if pos == len(indices)-1:
        x[pos] = remaining
        evaluate_current()
        return
    for v in range(remaining+1):
        x[pos] = v
        dfs(pos+1, remaining - v)

# 目标函数计算
def evaluate_current():
    alloc = np.array(x)
    total_after = pred_df['现有警力数'].values + alloc
    ratio_dev = (total_after / np.where(baseline_pred==0, 1, baseline_pred) - 1)
    if np.isnan(ratio_dev).any():
        # 发生异常，给一个大惩罚
        term1 = 1e9
    else:
        term1 = np.sum(ratio_dev**2)
    term2 = np.sum(k_i * alloc)
    F = W1 * term1 - W2 * term2
    best_solutions.append((F, alloc.copy()))

dfs(0, TOTAL_NEW)

best_solutions.sort(key=lambda t: t[0])

best_F, best_alloc = best_solutions[0]

# 前 TOP_K
topK = best_solutions[:TOP_K]

# ---------------- 输出结果 ----------------
# 回归摘要
with open(OUT_REG_SUMMARY, 'w', encoding='utf-8') as f:
    f.write('--- 回归模型摘要 ---\n')
    f.write(f'自变量(标准化后)：{features}\n')
    f.write('系数 (beta0 开始): ' + ', '.join(f'{b:.6f}' for b in beta) + '\n')
    f.write('\n均值: '+ ', '.join(f'{k}:{v:.3f}' for k,v in means.items())+'\n')
    f.write('标准差: '+ ', '.join(f'{k}:{v:.3f}' for k,v in stds.items())+'\n')
    # OLS 残差
    resid_reg = Y - Y_pred_reg
    f.write(f'样本数: {len(Y)}\n')
    f.write(f'OLS_RSS: {np.sum(resid_reg**2):.4f}\n')
    f.write(f'OLS_预测均值: {Y_pred_reg.mean():.3f}\n')
    f.write(f'OLS_残差Std: {resid_reg.std(ddof=1):.4f}\n')
    # 基线（可能是工作负荷 / ridge / OLS）
    resid_base = Y - baseline_pred
    f.write(f'Baseline 方法: {baseline_method}\n')
    f.write(f'Baseline_RSS: {np.sum(resid_base**2):.4f}\n')
    f.write(f'Baseline_预测均值: {baseline_pred.mean():.3f}\n')
    f.write(f'Baseline_残差Std: {np.std(resid_base, ddof=1):.4f}\n')
    f.write(f'缺编总数(基线取整): {D.sum()}\n')
    if USE_WORKLOAD_METHOD and workload_capacity_used is not None:
        f.write(f'工作负荷容量(裁剪后中位数): {workload_capacity_used:.4f}\n')
    if USE_WORKLOAD_METHOD and workload_scale_used is not None:
        f.write(f'工作负荷预测缩放系数: {workload_scale_used:.4f}\n')
    if USE_WORKLOAD_METHOD and workload_original_shortage_rate is not None:
        f.write(f'工作负荷原始缺编率: {workload_original_shortage_rate:.4%}\n')
    if USE_WORKLOAD_METHOD and workload_final_shortage_rate is not None:
        f.write(f'工作负荷最终缺编率: {workload_final_shortage_rate:.4%}\n')
    if USE_WORKLOAD_METHOD and workload_auto_scale_used is not None:
        f.write(f'工作负荷二分scale: {workload_auto_scale_used:.4f}\n')

pred_df.to_csv(OUT_PRED, index=False, encoding='utf-8-sig')

# 最优方案输出
alloc_best_df = pred_df[['Index','现有警力数','预测警力_取整','缺编人数D','k_i','采用方法']].copy()
alloc_best_df['分配x_i'] = best_alloc
alloc_best_df['分配后警力'] = alloc_best_df['现有警力数'] + alloc_best_df['分配x_i']
alloc_best_df['偏差平方'] = ((alloc_best_df['分配后警力'] / np.where(alloc_best_df['预测警力_取整']==0,1,alloc_best_df['预测警力_取整']) - 1)**2)
alloc_best_df.to_csv(OUT_ALLOC_BEST, index=False, encoding='utf-8-sig')

# 前K方案
rows = []
for rank,(F, alloc) in enumerate(topK, start=1):
    for i,idx in enumerate(indices):
        rows.append({'方案rank':rank,'F':F,'Index':idx,'x_i':alloc[i]})
alloc_top_df = pd.DataFrame(rows)
alloc_top_df.to_csv(OUT_ALLOC_TOP, index=False, encoding='utf-8-sig')

print('回归&优化完成 | 基线方法:', baseline_method)
print('最佳目标值 F =', best_F)
print('输出文件:')
print('  ', OUT_REG_SUMMARY)
print('  ', OUT_PRED)
print('  ', OUT_ALLOC_BEST)
print('  ', OUT_ALLOC_TOP)
print('\n[回归系数 beta (OLS 全特征)] (beta0 开始):')
for i,b in enumerate(beta):
    name = 'Intercept' if i==0 else features[i-1]
    print(f'  {name}: {b:.4f}')
if Y_pred_ridge is not None:
    print('[INFO] 已计算 Ridge 预测 (alpha=5.0)，用于备选。')

cols_show = [c for c in ['Index','现有警力数','预测_工作负荷_ceil','预测_Ridge_ceil','预测_OLS_ceil','预测警力_取整','缺编人数D','k_i','采用方法'] if c in pred_df.columns]
print('\n[预测与缺编概览]:')
print(pred_df[cols_show].to_string(index=False))

print('\n[最优分配方案 allocation_best]:')
print(alloc_best_df[['Index','现有警力数','预测警力_取整','分配x_i','分配后警力','偏差平方','k_i','采用方法']].to_string(index=False))

print('\n[前5方案摘要]')
for rank,(F, alloc) in enumerate(topK, start=1):
    print(f'  方案{rank}: F={F:.6f}; 分配向量=' + ','.join(str(a) for a in alloc))

print('\n打印完成。')
