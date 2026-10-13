import os
import json
import pandas as pd
import csv
import unicodedata

# ANSI 转义序列，用于在终端中输出带颜色的文本
RED = '\033[91m'
RESET = '\033[0m'  # 重置颜色

# 尝试将路径的字符标准化为文件系统兼容的格式
def normalize_path(path):
    # 规范化路径，去除非标准字符
    return unicodedata.normalize('NFC', path)

def save_df_to_tsv(dataframe, path):
    dataframe.to_csv(
        path,
        sep="\t",
        header=True,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )

def search_alternate_paths(file_path):
    # 在多个目录中尝试查找文件，作为简单示例：
    alternate_dirs = [
        '/root/autodl-tmp/backup/data/singing',
    ]
    for alt_dir in alternate_dirs:
        alt_path = file_path.replace('/root/autodl-tmp/data/singing', alt_dir)
        if os.path.exists(alt_path):
            return alt_path
    return None

def generate():
    input_file = '/root/autodl-tmp/data/singing/meta/metadata.json'
    output_file = '/root/autodl-tmp/tcsinger2/data/new/data.tsv'
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取所有可能的键，假设所有条目的键都相同
    all_keys = set()
    for item in data:
        all_keys.update(item.keys())
    all_keys = list(all_keys)
    
    # 将 'wav_fn' 替换为 'audio_path'
    all_keys.append('mel_path')
    if 'wav_fn' in all_keys:
        all_keys.remove('wav_fn')
        all_keys.append('audio_path')  # 使用 'audio_path' 代替 'wav_fn'

    manifest = {key: [] for key in all_keys}

    missing_files = []  # 用于记录不存在的音频文件

    print(f'总共{len(data)}')
    for item in data:
        # 提取所有属性
        tmp = {}
        value = item.get('wav_fn', '')
        
        # 对路径进行规范化处理
        value = normalize_path(value)
        
        # 检查 audio_path 是否存在
        if not os.path.exists(value):
            missing_files.append(value)  # 记录不存在的文件
            continue
        
        # 对 mel_path 路径进行类似的处理
        mel_value = item.get('wav_fn', '').replace('.wav', '_mel.npy')
        mel_value = mel_value.replace('/root/autodl-tmp/data/singing', '/root/autodl-tmp/tcsinger2/processed/melnone48000/singing').replace('nips_final/nips_submit', 'GTSinger')
        
        # 对 mel_path 路径进行规范化处理
        mel_value = normalize_path(mel_value)
        
        for key in all_keys:
            if key == 'mel_path':
                value = mel_value
            elif key == 'audio_path':
                value = item.get('wav_fn', '')
                value = normalize_path(value)  # 对音频路径进行规范化处理
            else:
                value = item.get(key, None)
            tmp[key] = value
            manifest[key].append(value)

    # 输出不存在的音频文件
    if missing_files:
        print(RED + f"以下音频文件不存在 ({len(missing_files)} 个):" + RESET)
        for f in missing_files:
            print(f)

    print(f"正在将清单写入 {output_file} ...")
    save_df_to_tsv(pd.DataFrame.from_dict(manifest), output_file)

if __name__ == '__main__':
    generate()
