"""
生成处理流程汇总报告
"""
import os
import pandas as pd
from datetime import datetime

def create_process_summary():
    """创建处理流程汇总表"""
    
    # 处理步骤信息
    process_steps = [
        {
            '步骤序号': 1,
            '步骤名称': '指标并集法对齐',
            '输入': '四个季度的原始CSV文件',
            '处理方法': '提取所有季度指标的并集，创建统一指标体系',
            '输出文件': '步骤1_指标并集法结果.csv',
            '关键作用': '确保所有指标都被纳入评价体系，避免信息丢失'
        },
        {
            '步骤序号': 2,
            '步骤名称': '数据对齐处理',
            '输入': '各季度原始数据',
            '处理方法': '按统一指标格式重组数据，缺失指标填充NaN',
            '输出文件': '步骤2_数据对齐结果.csv',
            '关键作用': '保证各季度数据结构一致，便于后续统一处理'
        },
        {
            '步骤序号': 3,
            '步骤名称': '缺失值处理',
            '输入': '对齐后的数据',
            '处理方法': '使用均值填充法处理缺失值',
            '输出文件': '步骤3_缺失值处理结果.csv',
            '关键作用': '确保数据完整性，避免计算错误'
        },
        {
            '步骤序号': 4,
            '步骤名称': '数据标准化',
            '输入': '填充后的完整数据',
            '处理方法': 'Min-Max标准化，将数据缩放到[0,1]区间',
            '输出文件': '步骤4_数据标准化结果.csv',
            '关键作用': '消除量纲影响，确保不同指标可比性'
        },
        {
            '步骤序号': 5,
            '步骤名称': '熵权法权重计算',
            '输入': '标准化后的数据',
            '处理方法': '基于信息熵理论客观计算指标权重',
            '输出文件': '步骤5_熵权法权重计算结果.csv',
            '关键作用': '客观确定各指标重要程度，避免主观偏见'
        },
        {
            '步骤序号': 6,
            '步骤名称': '综合绩效评分',
            '输入': '标准化数据+指标权重',
            '处理方法': '加权求和计算季度得分，平均计算年度得分',
            '输出文件': '步骤6_综合绩效得分结果.csv',
            '关键作用': '得出最终排名，为奖励决策提供科学依据'
        }
    ]
    
    # 创建处理流程表
    process_df = pd.DataFrame(process_steps)
    
    # 保存到build文件夹
    build_path = os.path.join(os.path.dirname(__file__), '..', 'build')
    os.makedirs(build_path, exist_ok=True)
    
    # 保存流程汇总表
    summary_path = os.path.join(build_path, '数据处理流程汇总表.csv')
    process_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    
    print("=" * 100)
    print("数据处理流程汇总表")
    print("=" * 100)
    print(process_df.to_string(index=False))
    print(f"\n流程汇总表已保存至: {summary_path}")
    
    # 创建文件检查报告
    create_file_check_report(build_path)
    
    return process_df

def create_file_check_report(build_path):
    """创建文件生成检查报告"""
    
    expected_files = [
        '步骤1_指标并集法结果.csv',
        '步骤2_数据对齐结果.csv', 
        '步骤3_缺失值处理结果.csv',
        '步骤4_数据标准化结果.csv',
        '步骤5_熵权法权重计算结果.csv',
        '步骤6_综合绩效得分结果.csv',
        '最终结果_前5名获奖名单.csv',
        '最终结果_前3名获奖名单.csv',
        '治安所长绩效排名结果.csv',
        '治安所长绩效排名结果_权重信息.csv'
    ]
    
    file_check = []
    total_size = 0
    
    for filename in expected_files:
        filepath = os.path.join(build_path, filename)
        exists = os.path.exists(filepath)
        
        if exists:
            size = os.path.getsize(filepath)
            total_size += size
            status = '✓ 已生成'
            size_kb = f"{size/1024:.1f} KB"
        else:
            status = '✗ 未生成'
            size_kb = '0 KB'
        
        file_check.append({
            '文件名': filename,
            '生成状态': status,
            '文件大小': size_kb,
            '文件路径': filepath
        })
    
    check_df = pd.DataFrame(file_check)
    
    print("\n" + "=" * 80)
    print("文件生成检查报告")
    print("=" * 80)
    print(check_df.to_string(index=False))
    print(f"\n总计生成文件大小: {total_size/1024:.1f} KB")
    
    # 保存检查报告
    check_path = os.path.join(build_path, '文件生成检查报告.csv')
    check_df.to_csv(check_path, index=False, encoding='utf-8-sig')
    print(f"文件检查报告已保存至: {check_path}")
    
    return check_df

if __name__ == "__main__":
    create_process_summary()
