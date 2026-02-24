import argparse
import os
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split

from hy.data_loader import load_sample_info, load_separate_cohorts
from hy.message import message_to_feishu
from hy.tab_files import aggregate_tab_file


def get_location(location):
    if location == "WORKING_DIR":
        return args.working_dir
    elif location == "MODEL_DATA":
        return args.model_data_dir
    elif location == "SCRIPT":
        return args.script_dir
    elif location == "REPORT":
        return os.path.join(args.working_dir, "results/3_FeatureReduction")
    else:
        return os.path.join(args.working_dir, "results")

def process_task(task_id, working_dir, script_dir):
    # 动态生成文件路径（例如 train_1.tab, train_2.tab, ...）
    tab_file = os.path.join(working_dir, f"train.tab")
    blacklist_file = os.path.join(script_dir, "blacklist.bed")
    genome_file = os.path.join(script_dir, "genome.txt")

    # 调用目标函数
    return aggregate_tab_file(tab_file, task_id, blacklist_file, genome_file)

def main(args):
    # p100 to p80 and p20
    sample_info = load_sample_info(get_location("MODEL_DATA"), 'dev')
    p80_ids_path = os.path.join(get_location("MODEL_DATA") + f"/{args.exp_name}.trn.ids.txt")
    # always generate p80_ids_path
    message_to_feishu(f"使用split_zr_ids.py生成的ids")
    p80 = load_separate_cohorts(get_location("MODEL_DATA"), args.exp_name, "trn")
    neg_ids = sample_info[(sample_info['target'] == 0) & (sample_info.index.isin(p80.index))].index
    p20_ids = load_separate_cohorts(get_location("MODEL_DATA"), args.exp_name, "neg")
    for i in range(1, 51):
        p64, _, _, _ = train_test_split(sample_info.loc[p80.index],
                                  sample_info.loc[p80.index]['target'],
                                  test_size=0.2,
                                  stratify=sample_info.loc[p80.index]['target'],
                                  random_state=i+1234)
        # 获取 p64 中 target=0 和 target=1 的数量
        count_0 = (p64['target'] == 0).sum()
        count_1 = (p64['target'] == 1).sum()

        # 取较少的一个类别数量
        min_count = min(count_0, count_1)

        # 对 target=0 和 target=1 分别采样 min_count 个样本
        balanced_p64 = pd.concat([
            p64[p64['target'] == 0].sample(n=min_count, random_state=i + 1234),
            p64[p64['target'] == 1].sample(n=min_count, random_state=i + 1234)
        ])

        with open(get_location("WORKING_DIR")+f"/all.{args.exp_name}.sample.info.{i}", 'w') as f:
            f.write(f"{len(balanced_p64)}\n")
            for index, row in balanced_p64.iterrows():
                f.write(f"{index} {row['target']} - -2 0 -1\n")
    # 生成 train.tab 文件
    message_to_feishu(f"开始生成 train.tab 文件")
    script_dir = get_location("SCRIPT")
    model_data_dir = get_location("MODEL_DATA")
    # 这里是为了生成train.tab文件
    os.system(f"bash {script_dir}/make_tab_fast.sh {model_data_dir}/{args.exp_name}.trn.ids.txt trim_q30_gcc_10k_cpm train.tab")
    # 并行处理 1-100 的任务
    r_enriched_col = 'r_enriched_value'
    r_depleted_col = 'r_depleted_value'
    message_to_feishu(f"生成info.csv，提取 {r_enriched_col} 和 {r_depleted_col} 列")
    sample_info_for_csv = sample_info.dropna(subset=[r_enriched_col, r_depleted_col])[[r_enriched_col, r_depleted_col]]
    sample_info_for_csv['label'] = 'test'
    sample_info_for_csv.loc[neg_ids, 'label'] = 'train'
    train_indices = p20_ids.index
    test_indices = sample_info_for_csv.index.difference(p20_ids.index)
    sample_info_for_csv_reordered = sample_info_for_csv.loc[train_indices.append(test_indices)]
    sample_info_for_csv_reordered.to_csv(get_location("WORKING_DIR") + f"/info.csv", index=True, header=True, columns=['label', r_enriched_col, r_depleted_col])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", help="实验名称 (如 gc hcc)")
    parser.add_argument('working_dir', help='工作目录')
    parser.add_argument('script_dir', help='脚本目录')
    parser.add_argument('model_data_dir', help='模型数据目录')

    args = parser.parse_args()

    start_time = datetime.now()
    main(args)
    end_time = datetime.now()
    elapsed_time = end_time - start_time

    print(f"程序运行时间: {elapsed_time}")
