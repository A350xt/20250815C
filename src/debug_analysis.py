"""
调试版本的数据处理脚本
"""
import sys
import os
import pandas as pd

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(__file__))

from data_processing import PerformanceDataProcessor

def debug_data_loading():
    """调试数据加载过程"""
    base_path = os.path.join(os.path.dirname(__file__), '..', 'problem')
    
    file_paths = {
        'Q1': os.path.join(base_path, 'C题：附件1-治安所长业务绩效第一季度.csv')
    }
    
    processor = PerformanceDataProcessor()
    
    # 先只加载第一季度数据进行调试
    print("=" * 50)
    print("调试数据加载过程")
    print("=" * 50)
    
    for quarter, file_path in file_paths.items():
        print(f"\n处理{quarter}季度数据: {file_path}")
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue
            
        # 读取原始文件内容
        print("\n原始文件前10行:")
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()[:10]):
                print(f"  {i+1}: {line.strip()}")
        
        # 使用处理器加载数据
        try:
            processor.load_quarterly_data({quarter: file_path})
            
            if quarter in processor.quarterly_data:
                df = processor.quarterly_data[quarter]
                print(f"\n处理后的数据形状: {df.shape}")
                print(f"列名: {list(df.columns)}")
                print(f"\n前5行数据:")
                print(df.head())
                
                # 检查数据类型
                print(f"\n数据类型:")
                for col in df.columns:
                    print(f"  {col}: {df[col].dtype}")
                
                # 检查数值列的统计信息
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    print(f"\n数值列统计信息:")
                    print(df[numeric_cols].describe())
            else:
                print("数据加载失败")
                
        except Exception as e:
            print(f"加载过程中出错: {e}")
            import traceback
            traceback.print_exc()

def run_full_debug():
    """运行完整的调试流程"""
    print("=" * 60)
    print("完整调试流程")
    print("=" * 60)
    
    processor = PerformanceDataProcessor()
    
    # 定义文件路径
    base_path = os.path.join(os.path.dirname(__file__), '..', 'problem')
    
    file_paths = {
        'Q1': os.path.join(base_path, 'C题：附件1-治安所长业务绩效第一季度.csv'),
        'Q2': os.path.join(base_path, 'C题^7附件1-治安所长业务绩效第二季度.csv'),
        'Q3': os.path.join(base_path, 'C题^L7附件1-治安所长业务绩效第三季度.csv'),
        'Q4': os.path.join(base_path, 'C题^LL7附件1-治安所长业务绩效第四季度.csv')
    }
    
    try:
        # 1. 加载数据
        print("\n1. 加载数据...")
        processor.load_quarterly_data(file_paths)
        
        if not processor.quarterly_data:
            print("没有成功加载任何数据")
            return
        
        # 2. 显示加载的数据概况
        print("\n2. 数据概况:")
        for quarter, df in processor.quarterly_data.items():
            print(f"  {quarter}: {df.shape[0]}行, {df.shape[1]}列")
            numeric_cols = df.select_dtypes(include=['number']).columns
            print(f"    数值列数量: {len(numeric_cols)}")
            print(f"    单位数量: {df['单位'].nunique()}")
            print(f"    分管领导数量: {df['分管领导'].nunique()}")
        
        # 3. 提取指标
        print("\n3. 提取统一指标...")
        indicators = processor.extract_unified_indicators()
        
        # 4. 对齐数据
        print("\n4. 对齐数据...")
        processor.align_data()
        
        # 5. 处理缺失值
        print("\n5. 处理缺失值...")
        processor.handle_missing_values(method='zero')  # 使用零填充便于调试
        
        # 6. 标准化（暂时跳过，先看原始数据）
        print("\n6. 跳过标准化，保持原始数值...")
        
        # 7. 计算权重
        print("\n7. 计算权重...")
        weights = processor.calculate_entropy_weights()
        
        # 8. 计算得分
        print("\n8. 计算综合得分...")
        scores = processor.calculate_comprehensive_scores()
        
        # 9. 显示结果
        print("\n9. 前10名结果:")
        top_10 = processor.get_top_performers(10)
        
    except Exception as e:
        print(f"调试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'load':
        debug_data_loading()
    else:
        run_full_debug()
