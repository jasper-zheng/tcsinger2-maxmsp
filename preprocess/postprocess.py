import pandas as pd
import ast
import json
import os
import re
from glob import glob

# 读取 TSV 文件
filename = './data/new/data.tsv'  # 输入文件路径
df = pd.read_csv(filename, sep='\t', header=0, low_memory=False)

# 需要填充"no"的列
fill_with_no_cols = ['emotion', 'range', 'pace', 'singing_method']

# tech类列
tech_cols = [
    'mix_tech', 'falsetto_tech', 'vibrato_tech', 'breathy_tech', 
    'pharyngeal_tech', 'weak_tech', 'bubble_tech', 'strong_tech', 'glissando_tech'
]

# 解析ph列
df['ph'] = df['ph'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)


def calculate_prompt_mel(mel_path, max_prompts=10):
    """
    根据 mel_path 计算多个 prompt_mel：
    1. 提取 _mel.npy 之前的数字部分。
    2. 扫描同目录下所有符合 _mel.npy 文件。
    3. 找到与目标数字最接近的多个路径（最多 max_prompts 个）。
    """

    if not isinstance(mel_path, str) or not os.path.exists(mel_path):
        print(f"mel file not found, skip: {mel_path}")
        return None

    if not isinstance(mel_path, str):
        assert False, f'Invalid input, mel_path is not a string: {mel_path}'

    base_dir = os.path.dirname(mel_path)  # 获取文件所在目录
    filename = os.path.basename(mel_path)  # 获取文件名

    # 正则匹配 _mel.npy 之前的数字部分
    match = re.search(r'(\d+)_mel\.npy$', filename)
    if not match:
        assert False, f'Invalid format, prompt mel not found for {mel_path}'

    original_number = int(match.group(1))  # 提取数字
    pattern = os.path.join(base_dir, '*_mel.npy')  # 匹配所有 _mel.npy 文件
    candidate_files = glob(pattern)  # 扫描符合条件的文件

    # 提取文件中的数字部分，并排除自己
    def extract_number(file):
        m = re.search(r'(\d+)_mel\.npy$', file)
        return int(m.group(1)) if m else None

    valid_files = [(file, extract_number(file)) for file in candidate_files]
    valid_files = [(file, num) for file, num in valid_files 
                   if num is not None and file != mel_path]

    if not valid_files:
        print(f'No valid mel files found in {base_dir} excluding {mel_path}')
        return None

    # 按数字与原始数字的距离排序
    valid_files.sort(key=lambda x: abs(x[1] - original_number))

    # 获取距离最接近的多个文件路径（最多 max_prompts 个）
    return [file for file, _ in valid_files[:max_prompts]]

for col in df.columns:
    if col == 'speech_fn':
        # 不填充
        continue
    elif col in fill_with_no_cols:
        # 空值填'no'
        df[col] = df[col].fillna('no')
        df.loc[df[col].eq(''), col] = 'no'
    elif col in tech_cols:
        def fill_tech(row):
            ph_len = len(row['ph'])
            val = row[col]
            if pd.isna(val) or val == '':
                return [2] * ph_len
            else:
                # 不为空则不做修改（假设用户确保长度足够）
                if isinstance(val, str):
                    val = ast.literal_eval(val)
                return val
        df[col] = df.apply(fill_tech, axis=1)
    else:
        # 对其他列空值不做处理
        pass

# 解析ph_durs字段
df['ph_durs'] = df['ph_durs'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# 计算总的duration列
df['duration'] = df['ph_durs'].apply(lambda ph_durs: sum(ph_durs) if isinstance(ph_durs, list) else 0)

# 去除duration大于20s的
df = df[df['duration'] <= 20]

# 去除ph超过100的
df = df[df['ph'].apply(lambda x: len(x) if isinstance(x, list) else 0) <= 100]

# 去掉ph小于2和duration小于2s的行
df = df[df['ph'].apply(lambda x: len(x) if isinstance(x, list) else 0) >= 2]
df = df[df['duration'] >= 2]

# 去掉包含 '#歌词' 的 item_name
df = df[~df['item_name'].str.contains('#歌词', na=False)]

# 过滤掉note_durs, ph_durs, ep_notedurs 中有负数的行
df = df[df['note_durs'].apply(lambda x: all(i >= 0 for i in ast.literal_eval(x)) if isinstance(x, str) else True)]
df = df[df['ph_durs'].apply(lambda x: all(i >= 0 for i in ast.literal_eval(x)) if isinstance(x, str) else True)]
df = df[df['ep_notedurs'].apply(lambda x: all(i >= 0 for i in ast.literal_eval(x)) if isinstance(x, str) else True)]

# 填充 language 列
def fill_language(row):
    # 如果 language 为空，则根据 item_name 填充
    if pd.isna(row['language']) or row['language'] == '':
        if '业余' in row['item_name'] or '专业' in row['item_name']:
            return 'English'  # 如果 item_name 包含 '业余' 或 '专业'，填充 English
        else:
            return 'Chinese'  # 否则填充 Chinese
    return row['language']  # 如果 language 不为空，直接返回原值

# 对 DataFrame 中的所有行应用填充语言的函数
df['language'] = df.apply(fill_language, axis=1)

# 增加technique列
if 'technique' not in df.columns:
    df['technique'] = ''
    for i, row in df.iterrows():
        techniques = []
        for col in tech_cols:
            if isinstance(row[col], list) and 1 in row[col]:
                techniques.append(col.split('_')[0])
        df.at[i, 'technique'] = ' and '.join(techniques) if techniques else 'no'

# 示例 DataFrame 和列操作
df['prompt_mel'] = df['mel_path'].apply(lambda x: calculate_prompt_mel(x, max_prompts=10))
df = df[df['prompt_mel'].apply(lambda x: x is not None and len(x) > 0)]  # 删除prompt_mel为空或没有找到路径的行

# 统计剩余数据的总duration
total_duration = df['duration'].sum()

# 打印统计结果
remaining_rows = df.shape[0]
print(f"Remaining rows after filtering: {remaining_rows}")
print(f"Total duration of remaining data: {total_duration} seconds")

# 保存处理后的数据
output_filename = './data/sing/new.tsv'
df.to_csv(output_filename, index=False, sep='\t')

print(f"Processed file saved to {output_filename}")

# 保存处理后的数据
output_filename = './data/sing/new.tsv'  # 输出文件路径
df.to_csv(output_filename, index=False, sep='\t')

