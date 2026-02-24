import os
from typing import List

import pandas as pd
from argparse import ArgumentParser

r_enriched_mode = 'r_enriched'
r_depleted_mode = "r_depleted"


def load_index(file_name: str) -> List[float]:
    """加载索引文件"""
    with open(file_name, 'r') as f:
        lines = f.readlines()

    # 跳过第一行（unset($lines[0]) 的等效操作）
    dict_list = []
    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            chr_name, start, end = parts[0], int(parts[1]), int(parts[2])
            dict_list.append((end - start) / 10000)

    return dict_list


def raw(id: str, type: str = "q30", origin: bool = False) -> List[str]:
    """读取原始数据文件"""
    dir_name = 'origin' if origin else 'cleaned'
    file_name = f"modelData/{type}/{dir_name}/{id}.raw"

    if not os.path.exists(file_name):
        return []

    with open(file_name, 'r') as f:
        return f.read().splitlines()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('id_file', type=str, default='ids.txt')
    parser.add_argument('sample_info_file', type=str, default='sampleinfo.dev.txt')

    args = parser.parse_args()
    id_file = args.id_file
    sample_info_file = args.sample_info_file
    # 加载索引文件
    r_enriched_index = load_index(f"modelData/{r_enriched_mode}/sorted.tab.index")
    r_depleted_index = load_index(f"modelData/{r_depleted_mode}/sorted.tab.index")

    # 读取 ID 文件
    with open(id_file, 'r') as f:
        ids = [line.strip() for line in f.readlines()]
    # 处理每个 ID
    rbcdna_info  = []
    for id in ids:

        r_enriched_data = raw(id, r_enriched_mode)
        r_enriched_normalized = []
        if r_enriched_data:  # 确保数据不为空
            for i in range(1, len(r_enriched_data)):
                r_enriched_normalized.append(float(r_enriched_data[i]) / r_enriched_index[i - 1])

        r_depleted_data = raw(id, r_depleted_mode)
        r_depleted_normalized = []
        if r_depleted_data:  # 确保数据不为空
            for i in range(1, len(r_depleted_data)):
                r_depleted_normalized.append(float(r_depleted_data[i]) / r_depleted_index[i - 1])

        # 计算总和并四舍五入
        r_enriched_sum = round(sum(r_enriched_normalized), 6) if r_enriched_normalized else 0
        r_depleted_sum = round(sum(r_depleted_normalized), 6) if r_depleted_normalized else 0
        rbcdna_info.append({
            'seqID': id,
            'r_enriched_value': r_enriched_sum,
            'r_depleted_value': r_depleted_sum,
        })
    rbcdna_info = pd.DataFrame(data=rbcdna_info).set_index('seqID')
    sample_info = pd.read_csv(sample_info_file, index_col=0, sep="\t")
    if 'r_enriched_value' in sample_info.columns:
        exit(0)

    sample_info = sample_info.join(rbcdna_info, how='left')
    sample_info.to_csv(sample_info_file, sep='\t')
