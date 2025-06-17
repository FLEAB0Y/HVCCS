import pandas as pd
import numpy as np
import os

def process_questionnaire_data():
    """
    从wjx.xlsx读取问卷数据，提取每道题的真实感、空间感、舒适感和总体得分的均值和方差，
    并将结果写入file_parameters.xlsx
    """
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 设置文件路径
    wjx_file_path = os.path.join(script_dir, "..", "wjx.xlsx")
    params_file_path = os.path.join(script_dir, "..", "file_parameters.xlsx")
    
    if not os.path.exists(wjx_file_path):
        print(f"错误：未找到文件 - {wjx_file_path}")
        return
    
    try:
        print(f"正在读取问卷数据文件 {wjx_file_path}...")
        wjx_df = pd.read_excel(wjx_file_path)
        
        # 获取数据行数，用于确定倒数第二行和倒数第一行
        num_rows = wjx_df.shape[0]
        mean_row_idx = num_rows - 2  # 倒数第二行是均值
        var_row_idx = num_rows - 1   # 倒数第一行是方差
        
        # 从第8列开始提取数据（索引为7）
        score_data = wjx_df.iloc[:, 7:]
        
        # 提取均值和方差行
        means = score_data.iloc[mean_row_idx].values
        variances = score_data.iloc[var_row_idx].values
        
        print(f"找到问卷评分列数：{len(score_data.columns)}")
        
        # 计算每道题的数据
        num_questions = 228
        dimensions = ["真实感", "空间感", "舒适感", "总体得分"]
        
        # 检查数据是否足够
        expected_cols = num_questions * 4
        if len(means) < expected_cols:
            print(f"警告：问卷数据列数 ({len(means)}) 少于预期 ({expected_cols})，可能导致部分题目数据缺失")
        
        # 读取目标文件
        if not os.path.exists(params_file_path):
            print(f"错误：未找到目标文件 - {params_file_path}")
            return
        
        print(f"正在更新参数文件 {params_file_path}...")
        
        # 读取现有数据，保留原始索引
        params_df = pd.read_excel(params_file_path)
        
        # 确保参数文件有足够的行
        if params_df.shape[0] < num_questions + 1:  # +1 是为了表头数据行
            # 添加更多的行
            new_rows = num_questions + 1 - params_df.shape[0]
            new_df = pd.DataFrame(index=range(new_rows))
            params_df = pd.concat([params_df, new_df], ignore_index=True)
        
        # 准备要添加的列名
        new_column_names = []
        for dimension in dimensions:
            new_column_names.append(f"{dimension}均值")
            new_column_names.append(f"{dimension}方差")
        
        # 确保params_df有足够的列，并直接使用正确的列名
        for col_name in new_column_names:
            if col_name not in params_df.columns:
                params_df[col_name] = ""
        
        # 写入每道题的数据 - 直接从第1行开始（索引为0）
        for q in range(num_questions):
            if q >= len(means) // 4:
                print(f"警告：问题 {q+1} 数据不存在，跳过")
                continue
                
            for d_idx, dimension in enumerate(dimensions):
                col_idx = q * 4 + d_idx
                
                # 获取列名
                col_name_mean = f"{dimension}均值"
                col_name_var = f"{dimension}方差"
                
                # 写入均值和方差 (第q行，索引为q)
                # 注意这里改为q而不是q+1，因为第0行就是第1题
                params_df.loc[q, col_name_mean] = means[col_idx]
                params_df.loc[q, col_name_var] = variances[col_idx]
        
        # 保存更新后的DataFrame到Excel文件
        params_df.to_excel(params_file_path, index=False)
        
        print(f"数据已成功写入 {params_file_path}")
        print(f"处理了 {num_questions} 道题的 {len(dimensions)} 个维度评分")
        
    except Exception as e:
        print(f"处理数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    process_questionnaire_data()