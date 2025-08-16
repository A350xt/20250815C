#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
派出所警力绩效考核数据处理系统
根据思路分析文档第88行之后的数据处理步骤进行处理
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

class PerformanceDataProcessor:
    def __init__(self, src_folder, output_folder):
        self.src_folder = src_folder
        self.output_folder = output_folder
        self.quarters = ['1.csv', '2.csv', '3.csv', '4.csv']
        self.quarter_names = ['第一季度', '第二季度', '第三季度', '第四季度']
        self.raw_data = {}
        self.processed_data = {}
        self.all_indicators = set()
        self.person_data = {}
        
        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)
    
    def read_quarterly_data(self):
        """步骤1: 读取四个季度的CSV文件"""
        print("=== 步骤1: 读取四个季度的CSV文件 ===")
        
        for i, quarter_file in enumerate(self.quarters):
            file_path = os.path.join(self.src_folder, quarter_file)
            try:
                # 读取CSV文件
                df = pd.read_csv(file_path, encoding='utf-8-sig', header=None)
                
                # 前两个季度的格式处理
                if i < 2:  # 第一、二季度
                    # 找到指标行和数据开始行
                    header_row = None
                    data_start_row = None
                    
                    for idx, row in df.iterrows():
                        if '单位' in str(row.iloc[0]) and '分管' in str(row.iloc[1]):
                            header_row = idx
                            data_start_row = idx + 1
                            break
                    
                    if header_row is not None:
                        # 提取指标名称（第一行的指标描述）
                        indicators = df.iloc[0, 2:].fillna('').astype(str).tolist()
                        # 提取列名（包含单位、分管领导等）
                        columns = df.iloc[header_row, :].fillna('').astype(str).tolist()
                        
                        # 构建完整列名
                        full_columns = columns[:2]  # 单位、分管领导
                        for j, indicator in enumerate(indicators):
                            if j < len(columns) - 2:
                                full_columns.append(f"{indicator}")
                            else:
                                full_columns.append(f"指标_{j+1}")
                        
                        # 提取数据部分
                        data_df = df.iloc[data_start_row:].reset_index(drop=True)
                        data_df.columns = full_columns[:len(data_df.columns)]
                        
                        # 清理数据
                        data_df = data_df.dropna(subset=[data_df.columns[0], data_df.columns[1]], how='all')
                        
                        self.raw_data[self.quarter_names[i]] = data_df
                        print(f"{self.quarter_names[i]}: 读取 {len(data_df)} 条记录，{len(data_df.columns)} 个指标")
                        
                        # 收集所有指标
                        self.all_indicators.update(data_df.columns[2:])
                
                else:  # 第三、四季度的格式处理
                    # 第三、四季度的数据结构不同
                    # 第1行（索引0）是标题
                    # 第2行（索引1）是指标描述
                    # 第3行（索引2）是最高得分
                    # 从第4行（索引3）开始是数据
                    
                    # 提取指标名称（第1行，从第3列开始）
                    indicators = df.iloc[1, 2:].fillna('').astype(str).tolist()
                    
                    # 构建列名
                    full_columns = ['单位', '分管领导']  # 前两列固定
                    for indicator in indicators:
                        if indicator.strip():  # 非空指标
                            full_columns.append(indicator)
                    
                    # 提取数据部分（从第4行开始）
                    data_df = df.iloc[3:].reset_index(drop=True)
                    
                    # 只保留有数据的列
                    valid_cols = min(len(full_columns), len(data_df.columns))
                    data_df = data_df.iloc[:, :valid_cols]
                    data_df.columns = full_columns[:valid_cols]
                    
                    # 清理数据：去除空行
                    data_df = data_df.dropna(subset=[data_df.columns[0], data_df.columns[1]], how='all')
                    # 去除完全空白的行
                    data_df = data_df[data_df.iloc[:, 0].notna() | data_df.iloc[:, 1].notna()]
                    
                    self.raw_data[self.quarter_names[i]] = data_df
                    print(f"{self.quarter_names[i]}: 读取 {len(data_df)} 条记录，{len(data_df.columns)} 个指标")
                    
                    # 收集所有指标
                    self.all_indicators.update(data_df.columns[2:])
                
            except Exception as e:
                print(f"读取 {quarter_file} 时出错: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"总共收集到 {len(self.all_indicators)} 个不同指标")
    
    def extract_indicators(self):
        """步骤2: 提取所有季度的指标列名，形成指标并集"""
        print("\n=== 步骤2: 提取指标并集 ===")
        
        # 将指标转换为列表并排序
        self.all_indicators = sorted(list(self.all_indicators))
        
        print(f"指标并集包含 {len(self.all_indicators)} 个指标:")
        for i, indicator in enumerate(self.all_indicators[:10], 1):
            print(f"{i:2d}. {indicator}")
        if len(self.all_indicators) > 10:
            print(f"... 还有 {len(self.all_indicators) - 10} 个指标")
    
    def align_data_by_person(self):
        """步骤3: 按人员对齐数据，处理人员在不同季度可能在不同单位的情况"""
        print("\n=== 步骤3: 按人员对齐数据 ===")
        
        # 收集所有人员信息
        all_persons = set()
        for quarter, df in self.raw_data.items():
            if len(df.columns) >= 2:
                persons = df.iloc[:, 1].dropna().astype(str)
                persons = persons[persons != '']
                all_persons.update(persons)
        
        all_persons = sorted(list(all_persons))
        print(f"发现 {len(all_persons)} 个不同的人员: {all_persons}")
        
        # 为每个人员建立跨季度的数据记录
        for person in all_persons:
            self.person_data[person] = {}
            
            for quarter, df in self.raw_data.items():
                if len(df.columns) >= 2:
                    # 查找该人员在这个季度的记录
                    person_records = df[df.iloc[:, 1].astype(str) == person]
                    
                    if not person_records.empty:
                        # 如果有多条记录（可能在不同单位），取第一条
                        record = person_records.iloc[0]
                        unit = record.iloc[0] if pd.notna(record.iloc[0]) else "未知单位"
                        
                        # 创建该人员在该季度的数据字典
                        quarter_data = {
                            '单位': unit,
                            '季度': quarter
                        }
                        
                        # 添加各项指标数据
                        for col_idx, col_name in enumerate(df.columns[2:], 2):
                            if col_idx < len(record):
                                value = record.iloc[col_idx]
                                quarter_data[col_name] = value
                        
                        self.person_data[person][quarter] = quarter_data
                    else:
                        # 该人员在这个季度没有记录
                        self.person_data[person][quarter] = {
                            '单位': None,
                            '季度': quarter
                        }
        
        print(f"完成 {len(self.person_data)} 个人员的数据对齐")
    
    def create_unified_dataset(self):
        """步骤4: 创建统一的数据集，处理缺失值"""
        print("\n=== 步骤4: 创建统一数据集并处理缺失值 ===")
        
        unified_data = []
        
        for person, quarters_data in self.person_data.items():
            for quarter, quarter_data in quarters_data.items():
                if quarter_data.get('单位') is not None:  # 只处理有数据的记录
                    row = {
                        '人员': person,
                        '季度': quarter,
                        '单位': quarter_data['单位']
                    }
                    
                    # 添加所有指标的数据
                    for indicator in self.all_indicators:
                        value = quarter_data.get(indicator, np.nan)
                        
                        # 清理数据
                        if pd.isna(value) or value == '' or str(value).strip() == '':
                            row[indicator] = np.nan
                        elif str(value) in ['不分管', '未分管', '不管理']:
                            row[indicator] = np.nan
                        else:
                            try:
                                # 尝试转换为数值
                                row[indicator] = float(value)
                            except:
                                row[indicator] = np.nan
                    
                    unified_data.append(row)
        
        self.unified_df = pd.DataFrame(unified_data)
        print(f"创建统一数据集: {len(self.unified_df)} 条记录，{len(self.unified_df.columns)} 个字段")
        
        # 统计缺失值情况
        missing_stats = self.unified_df.isnull().sum()
        missing_rate = (missing_stats / len(self.unified_df) * 100).round(2)
        
        print("\n指标缺失值统计（前10个）:")
        for indicator in self.all_indicators[:10]:
            if indicator in missing_stats:
                print(f"  {indicator}: {missing_stats[indicator]} 个缺失 ({missing_rate[indicator]}%)")
    
    def fill_missing_values(self):
        """步骤5: 填补缺失的指标值"""
        print("\n=== 步骤5: 填补缺失值 ===")
        
        # 对于数值型指标，使用该人员的其他季度均值填补
        for person in self.unified_df['人员'].unique():
            person_mask = self.unified_df['人员'] == person
            person_data = self.unified_df[person_mask]
            
            for indicator in self.all_indicators:
                if indicator in self.unified_df.columns:
                    # 计算该人员在其他季度的均值
                    person_values = person_data[indicator].dropna()
                    if len(person_values) > 0:
                        mean_value = person_values.mean()
                        # 填补该人员的缺失值
                        self.unified_df.loc[person_mask & self.unified_df[indicator].isna(), indicator] = mean_value
        
        # 剩余缺失值用全体均值填补
        for indicator in self.all_indicators:
            if indicator in self.unified_df.columns:
                if self.unified_df[indicator].isna().any():
                    global_mean = self.unified_df[indicator].mean()
                    if not pd.isna(global_mean):
                        self.unified_df[indicator].fillna(global_mean, inplace=True)
                    else:
                        self.unified_df[indicator].fillna(0, inplace=True)
        
        print("缺失值填补完成")
    
    def standardize_data(self):
        """步骤6: 对所有指标进行标准化处理"""
        print("\n=== 步骤6: 数据标准化 ===")
        
        # 选择数值型指标进行标准化
        numeric_indicators = []
        for indicator in self.all_indicators:
            if indicator in self.unified_df.columns:
                if self.unified_df[indicator].dtype in ['float64', 'int64']:
                    numeric_indicators.append(indicator)
        
        # Min-Max标准化
        self.standardized_df = self.unified_df.copy()
        
        for indicator in numeric_indicators:
            values = self.standardized_df[indicator]
            min_val = values.min()
            max_val = values.max()
            
            if max_val != min_val:
                self.standardized_df[indicator] = (values - min_val) / (max_val - min_val)
            else:
                self.standardized_df[indicator] = 0.5  # 如果所有值相同，设为0.5
        
        print(f"完成 {len(numeric_indicators)} 个指标的标准化")
    
    def calculate_entropy_weights(self):
        """步骤7: 使用熵权法计算指标权重"""
        print("\n=== 步骤7: 熵权法计算权重 ===")
        
        # 选择数值型指标
        numeric_indicators = []
        for indicator in self.all_indicators:
            if indicator in self.standardized_df.columns:
                if self.standardized_df[indicator].dtype in ['float64', 'int64']:
                    numeric_indicators.append(indicator)
        
        data_matrix = self.standardized_df[numeric_indicators].values
        m, n = data_matrix.shape
        
        # 避免log(0)的情况，将0值替换为很小的正数
        data_matrix = np.where(data_matrix == 0, 1e-10, data_matrix)
        
        # 计算比例矩阵
        proportion_matrix = data_matrix / data_matrix.sum(axis=0)
        
        # 计算信息熵
        entropy = -np.sum(proportion_matrix * np.log(proportion_matrix), axis=0) / np.log(m)
        
        # 计算权重
        diversity = 1 - entropy
        weights = diversity / diversity.sum()
        
        # 保存权重结果
        self.weights = pd.DataFrame({
            '指标': numeric_indicators,
            '熵值': entropy,
            '差异性系数': diversity,
            '权重': weights
        })
        
        print("权重计算完成，前10个指标权重:")
        for i in range(min(10, len(self.weights))):
            print(f"  {self.weights.iloc[i]['指标']}: {self.weights.iloc[i]['权重']:.4f}")
    
    def calculate_comprehensive_scores(self):
        """步骤8: 计算每个人员的综合绩效得分"""
        print("\n=== 步骤8: 计算综合绩效得分 ===")
        
        # 计算各季度得分
        numeric_indicators = self.weights['指标'].tolist()
        weights_dict = dict(zip(self.weights['指标'], self.weights['权重']))
        
        # 为每条记录计算加权得分
        self.standardized_df['季度得分'] = 0
        for indicator in numeric_indicators:
            if indicator in self.standardized_df.columns:
                self.standardized_df['季度得分'] += (
                    self.standardized_df[indicator] * weights_dict[indicator]
                )
        
        # 计算每个人员的年度综合得分
        annual_scores = []
        for person in self.standardized_df['人员'].unique():
            person_data = self.standardized_df[self.standardized_df['人员'] == person]
            
            # 计算年度平均得分
            annual_score = person_data['季度得分'].mean()
            
            # 获取该人员的单位信息（取最后一个季度的单位）
            units = person_data['单位'].dropna().tolist()
            final_unit = units[-1] if units else "未知单位"
            
            annual_scores.append({
                '人员': person,
                '最终单位': final_unit,
                '参与季度数': len(person_data),
                '年度综合得分': annual_score,
                '季度得分详情': person_data[['季度', '单位', '季度得分']].to_dict('records')
            })
        
        self.annual_scores_df = pd.DataFrame(annual_scores)
        self.annual_scores_df = self.annual_scores_df.sort_values('年度综合得分', ascending=False)
        
        print("综合得分计算完成")
        print(f"参与评价的人员数: {len(self.annual_scores_df)}")
        print(f"最高得分: {self.annual_scores_df['年度综合得分'].max():.4f}")
        print(f"最低得分: {self.annual_scores_df['年度综合得分'].min():.4f}")
        print(f"平均得分: {self.annual_scores_df['年度综合得分'].mean():.4f}")
    
    def save_results(self):
        """保存处理结果到build文件夹"""
        print("\n=== 保存处理结果 ===")
        
        # 1. 保存原始数据摘要
        with pd.ExcelWriter(os.path.join(self.output_folder, '数据处理结果.xlsx'), engine='openpyxl') as writer:
            # 年度综合得分排名
            self.annual_scores_df[['人员', '最终单位', '参与季度数', '年度综合得分']].to_excel(
                writer, sheet_name='年度综合得分排名', index=False
            )
            
            # 指标权重
            self.weights.to_excel(writer, sheet_name='指标权重', index=False)
            
            # 标准化后的详细数据
            detail_df = self.standardized_df[['人员', '季度', '单位', '季度得分']].copy()
            detail_df.to_excel(writer, sheet_name='季度得分详情', index=False)
        
        # 2. 保存统一数据集
        self.unified_df.to_csv(os.path.join(self.output_folder, '统一数据集.csv'), 
                              index=False, encoding='utf-8-sig')
        
        # 3. 保存标准化数据集
        self.standardized_df.to_csv(os.path.join(self.output_folder, '标准化数据集.csv'), 
                                   index=False, encoding='utf-8-sig')
        
        # 4. 保存人员跨季度追踪数据
        person_tracking = []
        for person, quarters_data in self.person_data.items():
            for quarter, quarter_data in quarters_data.items():
                if quarter_data.get('单位') is not None:
                    person_tracking.append({
                        '人员': person,
                        '季度': quarter,
                        '单位': quarter_data['单位']
                    })
        
        person_tracking_df = pd.DataFrame(person_tracking)
        person_tracking_df.to_csv(os.path.join(self.output_folder, '人员跨季度追踪.csv'), 
                                 index=False, encoding='utf-8-sig')
        
        print("所有结果已保存到build文件夹")
        print("生成的文件:")
        print("  - 数据处理结果.xlsx (包含排名、权重、详情)")
        print("  - 统一数据集.csv")
        print("  - 标准化数据集.csv")
        print("  - 人员跨季度追踪.csv")
    
    def run_complete_process(self):
        """运行完整的数据处理流程"""
        print("开始派出所警力绩效考核数据处理...")
        print("=" * 60)
        
        self.read_quarterly_data()
        self.extract_indicators()
        self.align_data_by_person()
        self.create_unified_dataset()
        self.fill_missing_values()
        self.standardize_data()
        self.calculate_entropy_weights()
        self.calculate_comprehensive_scores()
        self.save_results()
        
        print("=" * 60)
        print("数据处理完成！")
        print("\n前10名人员排名:")
        for i, row in self.annual_scores_df.head(10).iterrows():
            print(f"{i+1:2d}. {row['人员']} ({row['最终单位']}) - {row['年度综合得分']:.4f}分")


def main():
    # 设置文件路径
    src_folder = r"c:\Users\cfcsn\OneDrive\本科个人资料\数学建模暑假培训\20250815C\src"
    output_folder = r"c:\Users\cfcsn\OneDrive\本科个人资料\数学建模暑假培训\20250815C\build"
    
    # 创建处理器并运行
    processor = PerformanceDataProcessor(src_folder, output_folder)
    processor.run_complete_process()


if __name__ == "__main__":
    main()
