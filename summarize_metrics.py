import os
import argparse
import csv

# ================= 配置区域 =================
# 固定根目录
ROOT_DIR = "/mnt/qfg/Tate_qfg/AD-DINOv3/TESTING_ALL"
# ROOT_DIR = "/mnt/qfg/Tate_qfg/AD-DINOv3/TESTING_ALL/base"

# 数据集目录列表 (数组形式)
DATASET_LIST = ["mvtec", "mpdd", "btad"] 
# 如果有更多数据集，例如: ["mvtec", "mpdd", "visa", "other_dataset"]

# 输出结果的TXT文件名
OUTPUT_FILENAME = "metrics_summary.txt"
# ===========================================

def get_row_stats(row):
    """辅助函数：格式化单行数据字符串"""
    return f"epoch{row['epoch']} mean_auc={row['mean_auc']} mean_f1={row['mean_f1']}"

def process_csv_file(file_path):
    """读取并处理单个CSV文件，返回最佳AUC、最佳F1和最后一行的数据"""
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return "empty"

        # 寻找 Best AUC (将字符串转换为浮点数进行比较)
        best_auc_row = max(rows, key=lambda x: float(x['mean_auc']))
        
        # 寻找 Best F1
        best_f1_row = max(rows, key=lambda x: float(x['mean_f1']))
        
        # 获取 Final (最后一行)
        final_row = rows[-1]

        return {
            'auc_best': best_auc_row,
            'f1_best': best_f1_row,
            'final': final_row
        }
    except Exception as e:
        return f"error: {str(e)}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, default="learnablePrompt/20260108_193354", help="Path to metric txt file")
    args = ap.parse_args()
    
    # 目标时间戳 (这里填入你想要的时间戳)
    TARGET_TIMESTAMP = args.path

    # 构造输出文件的完整路径 (默认保存在当前脚本运行目录下)
    output_path = os.path.join(ROOT_DIR, TARGET_TIMESTAMP, OUTPUT_FILENAME)
    
    # 准备写入内容
    lines_to_write = []
    
    # 添加总体标题
    header = f"train_{TARGET_TIMESTAMP}:"
    lines_to_write.append(header)
    print(f"Processing {header} ...")

    for dataset in DATASET_LIST:
        # 拼接完整路径: ROOT / TIMESTAMP / DATASET / parsed_mean_metrics.csv
        csv_path = os.path.join(ROOT_DIR, TARGET_TIMESTAMP, dataset, "parsed_mean_metrics.csv")
        
        result = process_csv_file(csv_path)
        
        # 写入数据集名称 (缩进4空格)
        lines_to_write.append(f"    {dataset}:")
        
        if result is None:
            print(f"  [Warning] File not found: {csv_path}")
            lines_to_write.append("        File not found.")
        elif result == "empty":
            lines_to_write.append("        File is empty.")
        elif isinstance(result, str) and result.startswith("error"):
            lines_to_write.append(f"        Error reading file: {result}")
        else:
            # 格式化输出 (缩进8空格)
            auc_str = get_row_stats(result['auc_best'])
            f1_str = get_row_stats(result['f1_best'])
            final_str = get_row_stats(result['final'])
            
            lines_to_write.append(f"        auc_best: {auc_str}")
            lines_to_write.append(f"        f1_best: {f1_str}")
            lines_to_write.append(f"        final: {final_str}")
            
            print(f"  [Success] Processed {dataset}")

    # 将内容写入TXT文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines_to_write))
        f.write('\n') # 文件末尾加个换行

    print(f"\n统计完成！结果已保存至: {output_path}")
    print("-" * 30)
    # 打印文件内容预览
    print('\n'.join(lines_to_write))

if __name__ == "__main__":
    main()