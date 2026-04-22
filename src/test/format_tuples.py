import os
import ast

# 目标文件路径
file_path = r"D:\softmove\Downloads\新文件 2.txt"

def read_format_and_count(path):
    try:
        # 1. 读取原始数据
        with open(path, 'r', encoding='utf-8') as f:
            raw_content = f.read().strip()
        
        # 2. 解析数据 (使用 ast.literal_eval 安全解析列表)
        # 如果末尾有逗号导致解析失败，尝试处理一下
        if raw_content.endswith(','):
            raw_content = raw_content[:-1]
        if not raw_content.endswith(']'):
            raw_content += ']'
            
        data_list = ast.literal_eval(raw_content)
        
        # 3. 统计元素个数
        count = len(data_list)
        print(f"数据统计完成：共包含 {count} 个元素 (tuples)")
        
        # 4. 格式化写入：一个 tuple 一行
        with open(path, 'w', encoding='utf-8') as f:
            f.write("[\n")
            for i, tup in enumerate(data_list):
                # 直接将 tuple 转为字符串并写入一行
                f.write(f"    {tup}")
                if i < count - 1:
                    f.write(",")
                f.write("\n")
            f.write("]\n")
            
        print(f"格式化数据已重新写入: {path}")
        return count
        
    except Exception as e:
        print(f"处理失败: {e}")
        return None

if __name__ == "__main__":
    read_format_and_count(file_path)
