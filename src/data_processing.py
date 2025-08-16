"""
派出所治安所长绩效考核数据处理模块
使用并集法进行指标对齐和数据处理
"""

import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class PerformanceDataProcessor:
    """绩效数据处理器"""
    
    def __init__(self):
        self.quarterly_data = {}  # 存储各季度数据
        self.unified_indicators = []  # 统一指标集合
        self.processed_data = None  # 处理后的数据
        self.weights = {}  # 指标权重
        
    def load_quarterly_data(self, file_paths: Dict[str, str]):
        """
        加载各季度数据
        
        Args:
            file_paths: 季度文件路径字典，格式为 {'Q1': 'path1', 'Q2': 'path2', ...}
        """
        for quarter, file_path in file_paths.items():
            try:
                # 读取CSV文件，跳过前几行找到实际数据
                df = self._load_and_parse_csv(file_path)
                
                # 数据清洗
                df = self._clean_data(df)
                
                if len(df) > 0:
                    self.quarterly_data[quarter] = df
                    print(f"成功加载{quarter}季度数据，共{len(df)}行")
                else:
                    print(f"警告：{quarter}季度数据为空")
                
            except Exception as e:
                print(f"加载{quarter}季度数据失败: {e}")
                import traceback
                traceback.print_exc()
    
    def _load_and_parse_csv(self, file_path: str) -> pd.DataFrame:
        """
        加载并解析CSV文件，处理复杂的表头结构
        
        Args:
            file_path: 文件路径
            
        Returns:
            解析后的数据框
        """
        # 先读取所有行来找到数据开始的位置
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 找到包含"单位"的行作为表头
        header_row = -1
        for i, line in enumerate(lines):
            if '单位' in line and '分管' in line:
                header_row = i
                break
        
        if header_row == -1:
            raise ValueError(f"无法找到有效的表头行在文件 {file_path}")
        
        # 从表头行开始读取数据
        df = pd.read_csv(file_path, encoding='utf-8', skiprows=header_row)
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗
        
        Args:
            df: 原始数据框
            
        Returns:
            清洗后的数据框
        """
        # 删除空行
        df = df.dropna(how='all')
        
        # 重置列名，确保第一列为单位，第二列为分管领导
        df.columns = ['单位', '分管领导'] + [f'指标_{i}' for i in range(len(df.columns)-2)]
        
        # 过滤掉不是真实数据的行（如包含"本项最高得分"等的行）
        df = df[~df['单位'].astype(str).str.contains('本项|最高|得分|表一|表二|表三|表四|表五', na=False)]
        df = df[~df['分管领导'].astype(str).str.contains('本项|最高|得分|表一|表二|表三|表四|表五', na=False)]
        
        # 移除单位和分管领导列中的空值行
        df = df.dropna(subset=['单位', '分管领导'])
        
        # 过滤掉明显不是派出所名称的行
        df = df[df['单位'].astype(str).str.contains('派出所|所', na=False)]
        
        # 处理"不分管"等无效值，转换为NaN
        numeric_columns = df.columns[2:]  # 除了单位和分管领导的其他列
        for col in numeric_columns:
            df[col] = df[col].replace(['不分管', '不分', '不管', '-', ''], np.nan)
            # 尝试转换为数值类型
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 重置索引
        df = df.reset_index(drop=True)
        
        print(f"数据清洗完成，保留{len(df)}行有效数据")
        if len(df) > 0:
            print(f"单位示例: {df['单位'].iloc[0]}, 分管领导示例: {df['分管领导'].iloc[0]}")
        
        return df
    
    def extract_unified_indicators(self):
        """
        提取统一指标集合（并集法）
        """
        all_indicators = set()
        quarterly_indicators = {}
        
        for quarter, df in self.quarterly_data.items():
            # 获取除单位和分管领导外的所有列名
            indicators = list(df.columns[2:])
            quarterly_indicators[quarter] = indicators
            all_indicators.update(indicators)
            print(f"{quarter}季度指标数量: {len(indicators)}")
        
        self.unified_indicators = sorted(list(all_indicators))
        
        # 创建指标对比表
        print(f"\n统一指标集合数量: {len(self.unified_indicators)}")
        self._create_indicators_comparison_table(quarterly_indicators)
        
        return self.unified_indicators
    
    def _create_indicators_comparison_table(self, quarterly_indicators):
        """
        创建各季度指标对比表
        """
        # 创建指标对比表
        indicators_df = pd.DataFrame({
            '指标序号': range(1, len(self.unified_indicators) + 1),
            '统一指标名称': self.unified_indicators
        })
        
        # 为每个季度添加一列，标记该指标是否存在
        for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
            if quarter in quarterly_indicators:
                indicators_df[f'{quarter}是否存在'] = indicators_df['统一指标名称'].apply(
                    lambda x: '✓' if x in quarterly_indicators[quarter] else '✗'
                )
            else:
                indicators_df[f'{quarter}是否存在'] = '✗'
        
        # 计算每个指标在多少个季度中存在
        indicators_df['存在季度数'] = 0
        for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
            col_name = f'{quarter}是否存在'
            if col_name in indicators_df.columns:
                indicators_df['存在季度数'] += (indicators_df[col_name] == '✓').astype(int)
        
        print("\n" + "="*80)
        print("步骤1：指标并集法结果表")
        print("="*80)
        print(indicators_df.to_string(index=False))
        
        # 保存到build文件夹
        build_path = os.path.join(os.path.dirname(__file__), '..', 'build')
        os.makedirs(build_path, exist_ok=True)
        output_path = os.path.join(build_path, '步骤1_指标并集法结果.csv')
        indicators_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n指标对比表已保存至: {output_path}")
        
        # 统计信息
        total_indicators = len(self.unified_indicators)
        common_indicators = len(indicators_df[indicators_df['存在季度数'] == 4])
        print(f"\n指标统计:")
        print(f"  总指标数: {total_indicators}")
        print(f"  四个季度都存在的指标数: {common_indicators}")
        print(f"  部分季度存在的指标数: {total_indicators - common_indicators}")
        
        return indicators_df
    
    def align_data(self):
        """
        使用并集法对齐各季度数据
        """
        aligned_data = {}
        alignment_summary = []
        
        for quarter, df in self.quarterly_data.items():
            # 创建对齐后的数据框
            aligned_df = pd.DataFrame()
            aligned_df['单位'] = df['单位']
            aligned_df['分管领导'] = df['分管领导']
            
            # 统计信息
            original_indicators = len(df.columns) - 2
            missing_indicators = 0
            
            # 为每个统一指标添加列
            for indicator in self.unified_indicators:
                if indicator in df.columns:
                    aligned_df[indicator] = df[indicator]
                else:
                    # 缺失指标填充NaN
                    aligned_df[indicator] = np.nan
                    missing_indicators += 1
            
            aligned_data[quarter] = aligned_df
            alignment_summary.append({
                '季度': quarter,
                '原始指标数': original_indicators,
                '统一指标数': len(self.unified_indicators),
                '缺失指标数': missing_indicators,
                '数据行数': len(aligned_df)
            })
        
        self.quarterly_data = aligned_data
        
        # 创建数据对齐结果表
        self._create_alignment_summary_table(alignment_summary)
        print("数据对齐完成")
    
    def _create_alignment_summary_table(self, alignment_summary):
        """
        创建数据对齐结果表
        """
        summary_df = pd.DataFrame(alignment_summary)
        
        print("\n" + "="*60)
        print("步骤2：数据对齐结果表")
        print("="*60)
        print(summary_df.to_string(index=False))
        
        # 保存到build文件夹
        build_path = os.path.join(os.path.dirname(__file__), '..', 'build')
        os.makedirs(build_path, exist_ok=True)
        output_path = os.path.join(build_path, '步骤2_数据对齐结果.csv')
        summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n数据对齐结果表已保存至: {output_path}")
        
        return summary_df
    
    def handle_missing_values(self, method='mean'):
        """
        处理缺失值
        
        Args:
            method: 处理方法，'mean'(均值填充), 'zero'(零填充), 'median'(中位数填充)
        """
        missing_summary = []
        
        for quarter, df in self.quarterly_data.items():
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            # 统计缺失值
            missing_before = df[numeric_columns].isnull().sum().sum()
            
            if method == 'mean':
                df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
            elif method == 'zero':
                df[numeric_columns] = df[numeric_columns].fillna(0)
            elif method == 'median':
                df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
            
            # 对于仍然存在的NaN值，用0填充
            df[numeric_columns] = df[numeric_columns].fillna(0)
            
            missing_after = df[numeric_columns].isnull().sum().sum()
            
            missing_summary.append({
                '季度': quarter,
                '数值列数': len(numeric_columns),
                '处理前缺失值数': missing_before,
                '处理后缺失值数': missing_after,
                '处理方法': method
            })
        
        self._create_missing_values_summary_table(missing_summary)
        print(f"缺失值处理完成，使用方法: {method}")
    
    def _create_missing_values_summary_table(self, missing_summary):
        """
        创建缺失值处理结果表
        """
        summary_df = pd.DataFrame(missing_summary)
        
        print("\n" + "="*60)
        print("步骤3：缺失值处理结果表")
        print("="*60)
        print(summary_df.to_string(index=False))
        
        # 保存到build文件夹
        build_path = os.path.join(os.path.dirname(__file__), '..', 'build')
        os.makedirs(build_path, exist_ok=True)
        output_path = os.path.join(build_path, '步骤3_缺失值处理结果.csv')
        summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n缺失值处理结果表已保存至: {output_path}")
        
        return summary_df
    
    def standardize_data(self, method='min_max'):
        """
        数据标准化
        
        Args:
            method: 标准化方法，'min_max'(最小-最大标准化), 'z_score'(Z分数标准化)
        """
        standardization_summary = []
        
        for quarter, df in self.quarterly_data.items():
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            # 记录标准化前的统计信息
            before_stats = {
                'mean': df[numeric_columns].mean().mean(),
                'std': df[numeric_columns].std().mean(),
                'min': df[numeric_columns].min().min(),
                'max': df[numeric_columns].max().max()
            }
            
            if method == 'min_max':
                # 最小-最大标准化
                for col in numeric_columns:
                    col_min = df[col].min()
                    col_max = df[col].max()
                    if col_max != col_min:
                        df[col] = (df[col] - col_min) / (col_max - col_min)
                    else:
                        df[col] = 0
                        
            elif method == 'z_score':
                # Z分数标准化
                for col in numeric_columns:
                    col_mean = df[col].mean()
                    col_std = df[col].std()
                    if col_std != 0:
                        df[col] = (df[col] - col_mean) / col_std
                    else:
                        df[col] = 0
            
            # 记录标准化后的统计信息
            after_stats = {
                'mean': df[numeric_columns].mean().mean(),
                'std': df[numeric_columns].std().mean(),
                'min': df[numeric_columns].min().min(),
                'max': df[numeric_columns].max().max()
            }
            
            standardization_summary.append({
                '季度': quarter,
                '标准化方法': method,
                '标准化前均值': round(before_stats['mean'], 4),
                '标准化后均值': round(after_stats['mean'], 4),
                '标准化前标准差': round(before_stats['std'], 4),
                '标准化后标准差': round(after_stats['std'], 4),
                '标准化前范围': f"{round(before_stats['min'], 2)}~{round(before_stats['max'], 2)}",
                '标准化后范围': f"{round(after_stats['min'], 2)}~{round(after_stats['max'], 2)}"
            })
        
        self._create_standardization_summary_table(standardization_summary)
        print(f"数据标准化完成，使用方法: {method}")
    
    def _create_standardization_summary_table(self, standardization_summary):
        """
        创建数据标准化结果表
        """
        summary_df = pd.DataFrame(standardization_summary)
        
        print("\n" + "="*100)
        print("步骤4：数据标准化结果表")
        print("="*100)
        print(summary_df.to_string(index=False))
        
        # 保存到build文件夹
        build_path = os.path.join(os.path.dirname(__file__), '..', 'build')
        os.makedirs(build_path, exist_ok=True)
        output_path = os.path.join(build_path, '步骤4_数据标准化结果.csv')
        summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n数据标准化结果表已保存至: {output_path}")
        
        return summary_df
    
    def calculate_entropy_weights(self):
        """
        使用熵权法计算指标权重
        """
        # 合并所有季度的数据
        all_data = []
        for quarter, df in self.quarterly_data.items():
            quarter_data = df.copy()
            quarter_data['季度'] = quarter
            all_data.append(quarter_data)
        
        if not all_data:
            raise ValueError("没有可用的数据来计算权重")
        
        combined_data = pd.concat(all_data, ignore_index=True)
        numeric_columns = combined_data.select_dtypes(include=[np.number]).columns
        
        # 移除季度列（如果存在）
        numeric_columns = [col for col in numeric_columns if col != '季度']
        
        if len(numeric_columns) == 0:
            raise ValueError("没有找到数值型指标列")
        
        weights = {}
        
        for col in numeric_columns:
            try:
                # 获取该列的有效数值
                values = combined_data[col].dropna().values
                
                if len(values) == 0 or np.all(values == 0):
                    weights[col] = 0
                    continue
                
                # 确保所有值为正数（加上小常数避免负数和零）
                values = np.abs(values) + 1e-10
                
                # 计算比重
                proportions = values / np.sum(values)
                
                # 计算熵值
                entropy = -np.sum(proportions * np.log(proportions + 1e-10))
                
                # 计算信息效用值（差异系数）
                if entropy == 0:
                    diversity = 1
                else:
                    diversity = 1 - entropy / np.log(len(values))
                
                weights[col] = max(0, diversity)  # 确保权重为非负
                
            except Exception as e:
                print(f"计算指标 {col} 的权重时出错: {e}")
                weights[col] = 0
        
        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in weights.items()}
        else:
            # 如果所有权重为0，则平均分配
            self.weights = {k: 1 / len(weights) for k in weights.keys()}
        
        # 输出权重信息
        self._create_weights_summary_table()
        print("熵权法权重计算完成")
        
        return self.weights
    
    def _create_weights_summary_table(self):
        """
        创建权重计算结果表
        """
        # 按权重降序排列
        sorted_weights = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)
        
        weights_data = []
        for i, (indicator, weight) in enumerate(sorted_weights, 1):
            weights_data.append({
                '排序': i,
                '指标名称': indicator,
                '权重值': round(weight, 6),
                '权重百分比': f"{round(weight * 100, 2)}%",
                '重要性等级': self._get_importance_level(weight, sorted_weights)
            })
        
        weights_df = pd.DataFrame(weights_data)
        
        print("\n" + "="*80)
        print("步骤5：熵权法权重计算结果表")
        print("="*80)
        print("前20个重要指标:")
        print(weights_df.head(20).to_string(index=False))
        
        if len(weights_df) > 20:
            print(f"\n... 共{len(weights_df)}个指标，完整结果请查看保存的文件")
        
        # 保存到build文件夹
        build_path = os.path.join(os.path.dirname(__file__), '..', 'build')
        os.makedirs(build_path, exist_ok=True)
        output_path = os.path.join(build_path, '步骤5_熵权法权重计算结果.csv')
        weights_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n权重计算结果表已保存至: {output_path}")
        
        # 权重统计信息
        high_importance = len(weights_df[weights_df['重要性等级'] == '高'])
        medium_importance = len(weights_df[weights_df['重要性等级'] == '中'])
        low_importance = len(weights_df[weights_df['重要性等级'] == '低'])
        
        print(f"\n权重分布统计:")
        print(f"  高重要性指标: {high_importance}个")
        print(f"  中重要性指标: {medium_importance}个")
        print(f"  低重要性指标: {low_importance}个")
        
        return weights_df
    
    def _get_importance_level(self, weight, sorted_weights):
        """
        根据权重确定重要性等级
        """
        total_indicators = len(sorted_weights)
        max_weight = sorted_weights[0][1]
        
        if weight >= max_weight * 0.7:
            return '高'
        elif weight >= max_weight * 0.3:
            return '中'
        else:
            return '低'
    
    def calculate_comprehensive_scores(self):
        """
        计算综合绩效得分
        """
        scores = {}
        
        # 为每个季度计算得分
        for quarter, df in self.quarterly_data.items():
            quarter_scores = {}
            
            for idx, row in df.iterrows():
                unit = str(row['单位']).strip()
                leader = str(row['分管领导']).strip()
                key = f"{unit}_{leader}"
                
                # 计算加权得分
                score = 0
                valid_indicators = 0
                
                for indicator, weight in self.weights.items():
                    if indicator in df.columns:
                        value = row[indicator]
                        if pd.notna(value) and isinstance(value, (int, float)):
                            score += float(value) * weight
                            valid_indicators += 1
                
                # 只有当至少有一个有效指标时才记录得分
                if valid_indicators > 0:
                    quarter_scores[key] = score
                else:
                    quarter_scores[key] = 0
            
            scores[quarter] = quarter_scores
            print(f"{quarter}季度计算了{len(quarter_scores)}个得分")
        
        # 计算年度综合得分（四个季度平均）
        annual_scores = {}
        all_keys = set()
        for quarter_scores in scores.values():
            all_keys.update(quarter_scores.keys())
        
        print(f"总共找到{len(all_keys)}个唯一的人员")
        
        for key in all_keys:
            quarterly_scores = []
            for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
                if quarter in scores and key in scores[quarter]:
                    quarterly_scores.append(scores[quarter][key])
            
            if quarterly_scores:
                annual_scores[key] = np.mean(quarterly_scores)
            else:
                annual_scores[key] = 0
        
        # 转换为排序结果
        sorted_scores = sorted(annual_scores.items(), key=lambda x: x[1], reverse=True)
        
        self.comprehensive_scores = {
            'quarterly': scores,
            'annual': annual_scores,
            'ranked': sorted_scores
        }
        
        self._create_comprehensive_scores_table()
        
        print(f"综合得分计算完成，共{len(annual_scores)}个人员")
        print(f"最高分: {sorted_scores[0][1]:.6f}, 最低分: {sorted_scores[-1][1]:.6f}")
        
        return self.comprehensive_scores
    
    def _create_comprehensive_scores_table(self):
        """
        创建综合得分结果表
        """
        # 创建详细的得分表
        scores_data = []
        
        for i, (key, annual_score) in enumerate(self.comprehensive_scores['ranked'], 1):
            unit, leader = key.split('_', 1)
            
            # 获取各季度得分
            q1_score = self.comprehensive_scores['quarterly'].get('Q1', {}).get(key, 0)
            q2_score = self.comprehensive_scores['quarterly'].get('Q2', {}).get(key, 0)
            q3_score = self.comprehensive_scores['quarterly'].get('Q3', {}).get(key, 0)
            q4_score = self.comprehensive_scores['quarterly'].get('Q4', {}).get(key, 0)
            
            scores_data.append({
                '排名': i,
                '单位': unit,
                '分管领导': leader,
                'Q1得分': round(q1_score, 4),
                'Q2得分': round(q2_score, 4),
                'Q3得分': round(q3_score, 4),
                'Q4得分': round(q4_score, 4),
                '年度综合得分': round(annual_score, 6),
                '是否前五名': '★' if i <= 5 else '',
                '是否前三名': '★★' if i <= 3 else ''
            })
        
        scores_df = pd.DataFrame(scores_data)
        
        print("\n" + "="*120)
        print("步骤6：综合绩效得分结果表")
        print("="*120)
        print("前15名结果:")
        print(scores_df.head(15).to_string(index=False))
        
        if len(scores_df) > 15:
            print(f"\n... 共{len(scores_df)}名人员，完整结果请查看保存的文件")
        
        # 保存到build文件夹
        build_path = os.path.join(os.path.dirname(__file__), '..', 'build')
        os.makedirs(build_path, exist_ok=True)
        output_path = os.path.join(build_path, '步骤6_综合绩效得分结果.csv')
        scores_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n综合得分结果表已保存至: {output_path}")
        
        # 统计信息
        print(f"\n得分统计信息:")
        print(f"  平均得分: {scores_df['年度综合得分'].mean():.6f}")
        print(f"  标准差: {scores_df['年度综合得分'].std():.6f}")
        print(f"  最高得分: {scores_df['年度综合得分'].max():.6f}")
        print(f"  最低得分: {scores_df['年度综合得分'].min():.6f}")
        
        return scores_df
    
    def get_top_performers(self, top_n: int = 5):
        """
        获取前N名表现者
        
        Args:
            top_n: 前N名
            
        Returns:
            前N名的列表
        """
        if not hasattr(self, 'comprehensive_scores'):
            raise ValueError("请先计算综合得分")
        
        top_performers = self.comprehensive_scores['ranked'][:top_n]
        
        # 创建前N名结果表
        self._create_top_performers_table(top_performers, top_n)
        
        return top_performers
    
    def _create_top_performers_table(self, top_performers, top_n):
        """
        创建前N名表现者表格
        """
        top_data = []
        
        for i, (key, score) in enumerate(top_performers, 1):
            unit, leader = key.split('_', 1)
            
            # 获取各季度得分
            q1_score = self.comprehensive_scores['quarterly'].get('Q1', {}).get(key, 0)
            q2_score = self.comprehensive_scores['quarterly'].get('Q2', {}).get(key, 0)
            q3_score = self.comprehensive_scores['quarterly'].get('Q3', {}).get(key, 0)
            q4_score = self.comprehensive_scores['quarterly'].get('Q4', {}).get(key, 0)
            
            top_data.append({
                '名次': i,
                '单位': unit,
                '分管领导': leader,
                'Q1季度得分': round(q1_score, 4),
                'Q2季度得分': round(q2_score, 4),
                'Q3季度得分': round(q3_score, 4),
                'Q4季度得分': round(q4_score, 4),
                '年度综合得分': round(score, 6),
                '获奖类型': '前三名排序奖励' if i <= 3 else '前五名奖励'
            })
        
        top_df = pd.DataFrame(top_data)
        
        print(f"\n" + "="*100)
        print(f"最终结果：前{top_n}名治安所长获奖名单")
        print("="*100)
        print(top_df.to_string(index=False))
        
        # 保存到build文件夹
        build_path = os.path.join(os.path.dirname(__file__), '..', 'build')
        os.makedirs(build_path, exist_ok=True)
        output_path = os.path.join(build_path, f'最终结果_前{top_n}名获奖名单.csv')
        top_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n前{top_n}名获奖名单已保存至: {output_path}")
        
        return top_df
    
    def save_results(self, output_path: str):
        """
        保存处理结果
        
        Args:
            output_path: 输出文件路径
        """
        if not hasattr(self, 'comprehensive_scores'):
            raise ValueError("请先计算综合得分")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建结果数据框
        results = []
        for i, (key, score) in enumerate(self.comprehensive_scores['ranked'], 1):
            unit, leader = key.split('_', 1)
            results.append({
                '排名': i,
                '单位': unit,
                '分管领导': leader,
                '综合得分': round(score, 6),
                '是否前五名': '是' if i <= 5 else '否',
                '是否前三名': '是' if i <= 3 else '否'
            })
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        # 同时保存权重信息
        weights_path = output_path.replace('.csv', '_权重信息.csv')
        weights_df = pd.DataFrame([
            {'指标': k, '权重': round(v, 6)} 
            for k, v in sorted(self.weights.items(), key=lambda x: x[1], reverse=True)
        ])
        weights_df.to_csv(weights_path, index=False, encoding='utf-8-sig')
        
        print(f"排名结果已保存至: {output_path}")
        print(f"权重信息已保存至: {weights_path}")
        
        return results_df

def main():
    """主函数示例"""
    # 创建数据处理器
    processor = PerformanceDataProcessor()
    
    # 定义文件路径（相对于src文件夹，指向problem文件夹）
    import os
    base_path = os.path.join(os.path.dirname(__file__), '..', 'problem')
    
    file_paths = {
        'Q1': os.path.join(base_path, 'C题：附件1-治安所长业务绩效第一季度.csv'),
        'Q2': os.path.join(base_path, 'C题^7附件1-治安所长业务绩效第二季度.csv'),
        'Q3': os.path.join(base_path, 'C题^L7附件1-治安所长业务绩效第三季度.csv'),
        'Q4': os.path.join(base_path, 'C题^LL7附件1-治安所长业务绩效第四季度.csv')
    }
    
    # 数据处理流程
    print("=" * 50)
    print("派出所治安所长绩效考核数据处理")
    print("=" * 50)
    
    # 1. 加载数据
    processor.load_quarterly_data(file_paths)
    
    # 2. 提取统一指标
    processor.extract_unified_indicators()
    
    # 3. 对齐数据
    processor.align_data()
    
    # 4. 处理缺失值
    processor.handle_missing_values(method='mean')
    
    # 5. 数据标准化
    processor.standardize_data(method='min_max')
    
    # 6. 计算权重
    processor.calculate_entropy_weights()
    
    # 7. 计算综合得分
    processor.calculate_comprehensive_scores()
    
    # 8. 获取前五名
    print("\n" + "=" * 30)
    print("前五名治安所长（不排序奖励）")
    print("=" * 30)
    top_5 = processor.get_top_performers(5)
    
    # 9. 获取前三名
    print("\n" + "=" * 30)
    print("前三名治安所长（排序奖励）")
    print("=" * 30)
    top_3 = processor.get_top_performers(3)
    
    # 10. 保存结果
    # 保存到与src同级的build文件夹
    build_path = os.path.join(os.path.dirname(__file__), '..', 'build')
    os.makedirs(build_path, exist_ok=True)  # 如果build文件夹不存在则创建
    output_path = os.path.join(build_path, '治安所长绩效排名结果.csv')
    processor.save_results(output_path)

if __name__ == "__main__":
    main()
