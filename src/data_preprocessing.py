"""附件2 & 附件3 数据预处理脚本
步骤概述：
1. 解析附件2（多行表头）生成结构化列名
2. 提取所需特征 + 生成人口、警情、案件三年均值特征
3. 解析附件3，稳健处理“事件平均花费时间”并计算总工作负荷时长
4. 合并、标准化并输出处理结果
输出：
  build/problem2/processed_data_raw.csv 原始特征
  build/problem2/processed_data_scaled.csv 标准化自变量
  build/problem2/feature_columns.txt 特征说明
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.preprocessing import StandardScaler

BASE = Path(r'C:/Users/cfcsn/OneDrive/本科个人资料/数学建模暑假培训/20250815C')
file2 = BASE / 'problem' / '附件2.csv'
file3 = BASE / 'problem' / '附件3.csv'
out_dir = BASE / 'build' / 'problem2'
out_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. 读取附件2（多行表头处理）
# --------------------------------------------------
raw2 = pd.read_csv(file2, header=None, dtype=str)

# 找到数据起始行（Index 形如 A0,A1...）
data_start_idx = None
for i in range(len(raw2)):
    v = str(raw2.iloc[i,0])
    if re.fullmatch(r'A\d+', v):
        data_start_idx = i
        break
if data_start_idx is None:
    raise RuntimeError('未找到数据起始行 (A0...)')

header_rows = raw2.iloc[:data_start_idx].fillna('')
data_rows = raw2.iloc[data_start_idx:].reset_index(drop=True)

# 组装列名（逐列拼接非空层级，去重）
col_names = []
for col in range(raw2.shape[1]):
    parts = []
    for r in range(header_rows.shape[0]):
        cell = str(header_rows.iloc[r, col]).strip()
        if cell and cell != 'nan':
            parts.append(cell)
    # 如果全为空，用占位名
    if not parts:
        col_name = f'COL{col}'
    else:
        col_name = '_'.join(parts)
    col_names.append(col_name)

# 首行是数据，不是表头
data_rows.columns = col_names

# 删除“分局总数”行
data_rows = data_rows[~data_rows.iloc[:,0].astype(str).str.contains('分局总数')]

# 将第一列命名为 Index
data_rows.rename(columns={data_rows.columns[0]:'Index'}, inplace=True)

# 去除可能的空白列（全空或全 NaN）
empty_cols = [c for c in data_rows.columns if data_rows[c].replace('', np.nan).isna().all()]
data_rows.drop(columns=empty_cols, inplace=True)

# 数值化(能转就转)
for c in data_rows.columns:
    if c != 'Index':
        data_rows[c] = pd.to_numeric(data_rows[c], errors='ignore')

# ---------------- 选择/构造特征 ----------------
# 基础列定位函数
def find_col(patterns, exclude=None, allow_multiple=False):
    pats = patterns if isinstance(patterns, (list,tuple,set)) else [patterns]
    candidates = []
    for c in data_rows.columns:
        name = str(c)
        if exclude and any(ex in name for ex in (exclude if isinstance(exclude,(list,tuple,set)) else [exclude])):
            continue
        if all(p in name for p in pats):
            candidates.append(c)
    if not candidates:
        return None if allow_multiple else None
    if allow_multiple:
        return candidates
    return candidates[0]

# 1) 区域面积 / 行业场所数
col_area = find_col('区域面积')
col_industry = find_col('行业场所数')

# 2) 人口（取2018年，若找不到则退化到 2016-2018三年均数）
def pick_population(year_key: str, pop_key: str):
    # 优先：包含 year_key & pop_key；备用：三年均数 & pop_key
    c_year = find_col([year_key, pop_key])
    if c_year: return c_year
    return find_col(['2016-2018三年均数', pop_key])

col_perm_pop = pick_population('2018年','常口')  # 常住人口
col_temp_pop = pick_population('2018年','暂口')  # 暂住人口
col_reg_pop  = pick_population('2018年','寄口')  # 寄住人口

# 3) 110警情数（单列）
col_110 = find_col('110警情数')
# 4) 刑事案件立案数（取三年均数）
col_crime_avg = find_col(['刑事案件立案数','三年']) or find_col('刑事案件立案数')
# 5) 治安案件查处数（取三年均数）
col_security_avg = find_col(['治安案件查处数','三年']) or find_col('治安案件查处数')

# 6) 警力（因变量 Y）第一列对应“警力数（编制）”所在原列：根据观察一般是第二列（Index 后）
col_force = find_col('警力数')
if col_force is None:
    # 回退：选数值列中与 38,48 等类似范围的列（启发式）
    numeric_cols = [c for c in data_rows.columns if c!='Index' and pd.api.types.is_numeric_dtype(data_rows[c])]
    col_force = numeric_cols[0]

feature_map = {
    'Index':'Index',
    '区域面积': col_area,
    '行业场所数': col_industry,
    '常住人口': col_perm_pop,
    '暂住人口': col_temp_pop,
    '寄住人口': col_reg_pop,
    '平均110警情数': col_110,
    '平均刑事案件立案数': col_crime_avg,
    '平均治安案件查处数': col_security_avg,
    '现有警力数': col_force,
}

missing = {k:v for k,v in feature_map.items() if v is None}
if missing:
    print('警告：以下特征列未匹配成功，将以0填充 ->', missing)

# --------------------------------------------------
# 补充：收集所有年份人口列并加入特征（常/暂/寄 2016-2021年）
# 列名格式中包含 年份 + 关键字(常口/暂口/寄口)
# 新列命名：常住人口_2016, 暂住人口_2017 等
# 追加：各人口类型多年均值列：常住人口_年均 等
# --------------------------------------------------
year_re = re.compile(r'20(16|17|18|19)年')
pop_key_map = {'常口':'常住人口','暂口':'暂住人口','寄口':'寄住人口'}
population_year_map = []  # (new_name, original_col)
population_year_groups = {v:[] for v in pop_key_map.values()}
for col in data_rows.columns:
    if col == 'Index':
        continue
    col_str = str(col)
    m = year_re.search(col_str)
    if not m:
        continue
    year = m.group(0)[:4]
    for raw_key, human in pop_key_map.items():
        if raw_key in col_str:
            new_name = f'{human}_{year}'
            population_year_map.append((new_name, col))
            population_year_groups[human].append(new_name)
            break

# 将年份人口列加入 feature_map（不覆盖已有基准列）
for new_name, orig_col in population_year_map:
    if new_name not in feature_map:
        feature_map[new_name] = orig_col
print(f"捕获年份人口列 {len(population_year_map)} 个")
for new_name, orig_col in population_year_map:
    print('  ', new_name, '<-', orig_col)

feat_df = pd.DataFrame()
feat_df['Index'] = data_rows['Index']
for new_name, old_name in feature_map.items():
    if new_name=='Index':
        continue
    if old_name is None:
        feat_df[new_name] = 0
    else:
        feat_df[new_name] = pd.to_numeric(data_rows[old_name], errors='coerce')

# 多年均值列
for human, cols_list in population_year_groups.items():
    if cols_list:
        feat_df[f'{human}_年均'] = feat_df[cols_list].mean(axis=1)

# 景区哑变量
feat_df['is_scenic_spot'] = feat_df['Index'].isin(['A3','A9','A10']).astype(int)

# 缺失人口填 0
for c in ['常住人口','暂住人口','寄住人口']:
    feat_df[c] = feat_df[c].fillna(0)

# --------------------------------------------------
# 2. 解析附件3 计算工作负荷
# --------------------------------------------------
df3 = pd.read_csv(file3, dtype=str)
df3.rename(columns={df3.columns[0]:'Index'}, inplace=True)

# 转数值
for c in df3.columns:
    if c!='Index':
        df3[c] = pd.to_numeric(df3[c], errors='coerce')

# 所有“事件平均花费时间”列
avg_cols = [c for c in df3.columns if '事件平均花费时间' in c]

# 对每个平均时间列做异常值稳健缩尾后再取行内中位数
def winsor_series(s: pd.Series, k=1.5):
    q1, q3 = s.quantile([0.25,0.75])
    iqr = q3 - q1
    lo, hi = q1 - k*iqr, q3 + k*iqr
    return s.clip(lo, hi)

if avg_cols:
    df3[avg_cols] = df3[avg_cols].apply(winsor_series, axis=0)
    df3['稳健平均处理时间'] = df3[avg_cols].median(axis=1, skipna=True)
else:
    df3['稳健平均处理时间'] = np.nan

# 事件数量列（启发式：包含 “数” 但不包含 “平均” 且不是合计、不是占比）
count_cols = [c for c in df3.columns if ('数' in c or '总和' in c) and '平均' not in c and '占比' not in c and c not in ['Index']]

# 计算工作负荷 = Σ count * 稳健平均处理时间 (粗粒度估计)
df3['总工作负荷时长'] = df3['稳健平均处理时间'].fillna(df3['稳健平均处理时间'].median()) * df3[count_cols].fillna(0).sum(axis=1)

workload = df3[['Index','总工作负荷时长']]
feat_df = feat_df.merge(workload, on='Index', how='left')
feat_df['总工作负荷时长'] = feat_df['总工作负荷时长'].fillna(feat_df['总工作负荷时长'].median())

# --------------------------------------------------
# 3. 标准化（仅自变量，不含现有警力数）
# --------------------------------------------------
base_cols = ['区域面积','行业场所数','常住人口','暂住人口','寄住人口']
year_pop_cols = sorted([c for c in feat_df.columns if re.match(r'(常住人口|暂住人口|寄住人口)_20(16|17|18|19|20|21)$', c)])
avg_pop_cols = [c for c in feat_df.columns if c.endswith('_年均')]
X_cols = base_cols + year_pop_cols + avg_pop_cols + [
    '平均110警情数','平均刑事案件立案数','平均治安案件查处数','总工作负荷时长','is_scenic_spot'
]
for c in X_cols:
    if c not in feat_df.columns:
        feat_df[c] = 0

scaler = StandardScaler()
scaled = feat_df.copy()
scaled_X = scaler.fit_transform(scaled[X_cols])
for i,c in enumerate(X_cols):
    scaled[c] = scaled_X[:,i]

# --------------------------------------------------
# 4. 输出
# --------------------------------------------------
raw_out = out_dir / 'processed_data_raw.csv'
scaled_out = out_dir / 'processed_data_scaled.csv'
feat_doc = out_dir / 'feature_columns.txt'

feat_df.to_csv(raw_out, index=False, encoding='utf-8-sig')
scaled.to_csv(scaled_out, index=False, encoding='utf-8-sig')

with open(feat_doc,'w',encoding='utf-8') as f:
    f.write('列说明:\n')
    for k,v in feature_map.items():
        if k=='Index':
            continue
        f.write(f'{k} <- {v}\n')
    f.write('新增: is_scenic_spot 景区哑变量; 总工作负荷时长 来自附件3; processed_data_scaled 仅自变量标准化\n')

############################################################
# 追加：严格按照“2预处理”说明，对附件3逐事件平均时间做稳健处理
# 重新读取附件3并构建事件级工作负荷（替代之前粗粒度方法）
############################################################

def robust_clean_avg_times(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    avg_positions = [i for i,c in enumerate(cols) if c == '事件平均花费时间']
    event_info = []  # (event_idx, time_col, count_col, avg_col)
    for idx, pos in enumerate(avg_positions, start=1):
        # 期望结构: total_time(col pos-2), count(col pos-1), avg(col pos)
        count_col = cols[pos-1] if pos-1 >=0 else None
        time_col  = cols[pos-2] if pos-2 >=0 else None
        avg_col   = cols[pos]
        if count_col is None or time_col is None:
            continue
        # 要求 count_col 名里含“数”或“警情”
        if ('数' in count_col or '警情' in count_col) and count_col not in ['Index']:
            event_info.append((f'E{idx}', time_col, count_col, avg_col, pos))

    # 复制一份用于清洗
    df_clean = df.copy()
    workload_components = []
    per_event_avg_cols = []

    for tag, time_col, count_col, avg_col, pos in event_info:
        series = pd.to_numeric(df_clean[avg_col], errors='coerce')
        # IQR 去异常（替换为列中位数）
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
        median = series.median()
        cleaned = series.where((series>=low)&(series<=high), median)
        new_avg_col = f'{tag}_avgTime'
        df_clean[new_avg_col] = cleaned
        per_event_avg_cols.append(new_avg_col)
        cnt = pd.to_numeric(df_clean[count_col], errors='coerce').fillna(0)
        comp_col = f'{tag}_workload'
        df_clean[comp_col] = cnt * cleaned
        workload_components.append(comp_col)

    if workload_components:
        df_clean['总工作负荷时长_refined'] = df_clean[workload_components].sum(axis=1)
    else:
        df_clean['总工作负荷时长_refined'] = np.nan

    # 仅保留必要列
    keep_cols = ['Index'] + per_event_avg_cols + workload_components + ['总工作负荷时长_refined']
    return df_clean[keep_cols], event_info

# 使用 refined 工作负荷替换之前的粗粒度列
df3_refined, event_meta = robust_clean_avg_times(df3)
feat_df = feat_df.drop(columns=['总工作负荷时长'], errors='ignore').merge(df3_refined[['Index','总工作负荷时长_refined']], on='Index', how='left')
feat_df.rename(columns={'总工作负荷时长_refined':'总工作负荷时长'}, inplace=True)
feat_df['总工作负荷时长'] = feat_df['总工作负荷时长'].fillna(feat_df['总工作负荷时长'].median())

# 重新标准化（自变量包含改进后的工作负荷）
scaled = feat_df.copy()
scaled_X = scaler.fit_transform(scaled[X_cols])
for i,c in enumerate(X_cols):
    scaled[c] = scaled_X[:,i]

# 输出改进结果（覆盖之前文件）
feat_df.to_csv(raw_out, index=False, encoding='utf-8-sig')
scaled.to_csv(scaled_out, index=False, encoding='utf-8-sig')

# 追加事件元数据文档
with open(feat_doc,'a',encoding='utf-8') as f:
    f.write('\n事件映射（附件3）：\n')
    for tag, time_col, count_col, avg_col, pos in event_meta:
        f.write(f'{tag}: time={time_col}, count={count_col}, avgColHeader={avg_col} (原序号{pos})\n')
    f.write('每类工作负荷 = count * 清洗后avg， 总工作负荷时长 = 各类工作负荷之和。\n')

print('--- 细化事件工作负荷已完成并覆盖输出 ---')
print('原始特征文件:', raw_out)
print('标准化文件:', scaled_out)
print('特征映射说明（含事件元数据）:', feat_doc)
print('完成。')
