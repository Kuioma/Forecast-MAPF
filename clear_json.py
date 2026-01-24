import socket

import json
import os

def load_and_repair_json(file_path, max_tries=4, backup=False):
    """
    尝试加载 JSON 文件。
    如果失败，从文件末尾向前查找最后一个 '}' 或 ']'，
    截断之后的内容，保存修复后的文件，并重试。
    
    Args:
        file_path (str): JSON 文件路径
        max_tries (int): 最大重试次数
        backup (bool): 是否在修复前备份原文件（推荐开启）
    
    Returns:
        dict or list: 成功解析的 JSON 数据
    """
    # 读取原始内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    for attempt in range(max_tries):
        start = len(content)
        try:
            data = json.loads(content)
            # ✅ 成功解析！如果内容被修改过，就写回修复后的版本
            if content != original_content:
                if backup and os.path.exists(file_path):
                    backup_path = file_path + ".bak"
                    if not os.path.exists(backup_path):
                        with open(file_path, 'rb') as src, open(backup_path, 'wb') as dst:
                            dst.write(src.read())
                        print(f"Backed up original file to {backup_path}")
                # 写回修复后的内容
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                # print(f"✅ Repaired and saved: {file_path}")
            return data
        except json.JSONDecodeError as e:
            # print(f"Attempt {attempt + 1}: JSON decode error at char {e.pos}")
            # 从末尾向前找最后一个 } 或 ]
            idx = 0
            for i in range(len(content) - 2, -1, -1):
                if content[i] == ']' and content[i-1] == '}':
                    idx = i + 1
                    break
            if idx <= 0:
                raise ValueError(f"Cannot find valid end marker '}}' or ']' in {file_path}")
            content = content[:idx]
            # print(f"Truncated to {idx} chars, retrying...")
        except Exception as e:
            raise e

    raise json.JSONDecodeError("Failed to repair JSON after max tries", content, 0)

import glob

def repair_all_json_files(folder_path):
    json_files = glob.glob(os.path.join(folder_path, "**/*.json"), recursive=True)
    for fp in json_files:
        try:
            data = load_and_repair_json(fp)
            # print(f"✅ {fp} is valid or repaired.")
        except Exception as e:
            print(f"❌ Failed to repair {fp}: {e}")

repair_all_json_files("/home/mapf-gpt/temp")