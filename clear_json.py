import socket

import json
import os

def load_and_repair_json(file_path, max_tries=4, backup=False):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    for attempt in range(max_tries):
        start = len(content)
        try:
            data = json.loads(content)
            if content != original_content:
                if backup and os.path.exists(file_path):
                    backup_path = file_path + ".bak"
                    if not os.path.exists(backup_path):
                        with open(file_path, 'rb') as src, open(backup_path, 'wb') as dst:
                            dst.write(src.read())
                        print(f"Backed up original file to {backup_path}")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                # print(f"✅ Repaired and saved: {file_path}")
            return data
        except json.JSONDecodeError as e:
            # print(f"Attempt {attempt + 1}: JSON decode error at char {e.pos}")
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

repair_all_json_files("temp")