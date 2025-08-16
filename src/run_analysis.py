"""
运行治安所长绩效考核数据处理
"""
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(__file__))

from data_processing import PerformanceDataProcessor

def run_analysis():
    """运行完整的数据分析流程"""
    print("=" * 60)
    print("派出所治安所长绩效考核数据处理系统")
    print("=" * 60)
    
    try:
        # 创建数据处理器
        processor = PerformanceDataProcessor()
        
        # 定义文件路径
        base_path = os.path.join(os.path.dirname(__file__), '..', 'problem')
        
        file_paths = {
            'Q1': os.path.join(base_path, 'C题：附件1-治安所长业务绩效第一季度.csv'),
            'Q2': os.path.join(base_path, 'C题^7附件1-治安所长业务绩效第二季度.csv'),
            'Q3': os.path.join(base_path, 'C题^L7附件1-治安所长业务绩效第三季度.csv'),
            'Q4': os.path.join(base_path, 'C题^LL7附件1-治安所长业务绩效第四季度.csv')
        }
        
        # 检查文件是否存在
        print("检查数据文件...")
        missing_files = []
        for quarter, file_path in file_paths.items():
            if not os.path.exists(file_path):
                missing_files.append(f"{quarter}: {file_path}")
        
        if missing_files:
            print("错误：以下文件不存在：")
            for file in missing_files:
                print(f"  - {file}")
            return False
        
        print("所有数据文件检查完毕，开始处理...")
        
        # 数据处理流程
        print("\n1. 加载季度数据...")
        processor.load_quarterly_data(file_paths)
        
        print("\n2. 提取统一指标集合...")
        processor.extract_unified_indicators()
        
        print("\n3. 使用并集法对齐数据...")
        processor.align_data()
        
        print("\n4. 处理缺失值...")
        processor.handle_missing_values(method='mean')
        
        print("\n5. 数据标准化...")
        processor.standardize_data(method='min_max')
        
        print("\n6. 计算熵权法权重...")
        weights = processor.calculate_entropy_weights()
        
        print("\n7. 计算综合绩效得分...")
        scores = processor.calculate_comprehensive_scores()
        
        print("\n" + "=" * 50)
        print("分析结果")
        print("=" * 50)
        
        # 问题1：前五名负责人（不排序）
        print("\n【问题1解答】前五名负责人（不排序奖励）")
        print("-" * 50)
        top_5 = processor.get_top_performers(5)
        
        # 问题2：前三名负责人（排序）
        print("\n【问题2解答】前三名负责人（排序奖励）")
        print("-" * 50)
        top_3 = processor.get_top_performers(3)
        
        print("\n" + "=" * 60)
        print("数学模型说明")
        print("=" * 60)
        print("1. 指标对齐模型：使用并集法合并四个季度的所有指标")
        print("2. 缺失值处理模型：使用均值填充法处理缺失数据")
        print("3. 数据标准化模型：使用Min-Max标准化消除量纲影响")
        print("4. 权重计算模型：使用熵权法客观确定指标权重")
        print("5. 综合评价模型：加权平均法计算季度得分，算术平均法计算年度得分")
        
        print("\n模型公式：")
        print("  熵权法公式：w_j = (1 - E_j) / Σ(1 - E_j)")
        print("  其中 E_j = -Σ(p_ij * ln(p_ij)) / ln(n)")
        print("  综合得分：S_i = Σ(w_j * x_ij)")
        print("  年度得分：S_annual = (S_Q1 + S_Q2 + S_Q3 + S_Q4) / 4")
        
        # 保存结果
        print("\n8. 保存结果到build文件夹...")
        build_path = os.path.join(os.path.dirname(__file__), '..', 'build')
        os.makedirs(build_path, exist_ok=True)
        output_path = os.path.join(build_path, '治安所长绩效排名结果.csv')
        results_df = processor.save_results(output_path)
        
        print("\n" + "=" * 50)
        print("处理完成！")
        print("=" * 50)
        print(f"结果文件保存在: {os.path.abspath(build_path)}")
        
        # 生成汇总报告
        print("\n生成处理流程汇总报告...")
        try:
            from create_summary import create_process_summary
            create_process_summary()
        except Exception as e:
            print(f"生成汇总报告时出错: {e}")
        
        return True
        
    except Exception as e:
        print(f"\n错误：数据处理过程中出现异常：{e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_analysis()
    if success:
        print("\n数据处理成功完成！")
    else:
        print("\n数据处理失败，请检查错误信息。")
