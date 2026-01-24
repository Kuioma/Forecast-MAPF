import json
with open('/home/mapf-gpt/table.content.txt', 'r') as file:
    table_text = file.read()
lines = table_text.strip().split('\n')
headers = [header.strip() for header in lines[1].split('|')[1:-1]]

# 处理每行数据
rows = []
for line in lines[3:-1]:
    values = [value.strip() for value in line.split('|')[1:-1]]
    row = {headers[i]: values[i] for i in range(len(headers))}
    rows.append(row)

# 将数据转换为JSON格式
json_data = json.dumps(rows, indent=4)

output_name = 'table_data.json'
# 保存为JSON文件
with open(output_name, 'w') as f:
    f.write(json_data)

print("数据已成功保存为 'table_data.json'")