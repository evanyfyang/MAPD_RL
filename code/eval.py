import csv
from collections import defaultdict

def process_file(input_file, output_file):
    # 使用defaultdict来存储数据，键为(agent_num, fre)，值为一个列表存储两次出现的值
    data = defaultdict(list)

    # 读取输入文件
    with open(input_file, 'r') as f:
        for line in f:
            agent_num, fre, value = line.strip().split()
            agent_num = int(agent_num)
            fre = float(fre)
            value = float(value)
            data[(agent_num, fre)].append(value)

    # 对数据进行排序：首先按frequency排序，然后按agent_number排序
    sorted_data = sorted(data.items(), key=lambda x: (x[0][1], x[0][0]))

    # 写入CSV文件
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['agent_number', 'frequency', 'with_delivering', 'without_delivering']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for (agent_num, fre), values in sorted_data:
            if len(values) == 2:
                writer.writerow({
                    'agent_number': agent_num,
                    'frequency': fre,
                    'with_delivering': values[0],
                    'without_delivering': values[1]
                })
            else:
                print(f"Warning: Expected 2 values for ({agent_num}, {fre}), but got {len(values)}")

# 使用示例
process_file('test_env_with_delivering.txt', 'output.csv')