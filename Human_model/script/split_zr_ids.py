import argparse
import os
from datetime import datetime

from sklearn.model_selection import train_test_split

from hy.data_loader import load_sample_info, load_separate_cohorts

def get_location(location):
    if location == "WORKING_DIR":
        return args.working_dir
    elif location == "MODEL_DATA":
        return os.path.join(args.working_dir, "modelData")
    elif location == "SCRIPT":
        return os.path.join(args.working_dir, "script")
    elif location == "REPORT":
        return os.path.join(args.working_dir, "results/3_FeatureReduction")
    else:
        return os.path.join(args.working_dir, "results")

def main(args):
    # p100 to p80 and p20
    base_name = args.exp_name

    sample_info = load_sample_info(get_location("MODEL_DATA"), 'dev')
    p100 = load_separate_cohorts(get_location("MODEL_DATA"), base_name, "p100")
    seed = 1234
    print((sample_info.loc[p100.index]['label'].value_counts()))
    sample_info['stratify_col'] = sample_info.loc[p100.index]['label'] + '_' + sample_info.loc[p100.index]['stage']
    print(sample_info['stratify_col'].value_counts())
    p80, p20, y_p80, y_p20 = train_test_split(sample_info.loc[p100.index],
                 sample_info.loc[p100.index]['target'],
                 test_size=0.2,
                 stratify=sample_info.loc[p100.index]['label'] + '_' + sample_info.loc[p100.index]['stage'],
                 random_state=seed)
    p80.to_csv(get_location("MODEL_DATA") + f"/{base_name}.trn.ids.txt", sep=',', index=True, header=False, columns=[])
    p20.to_csv(get_location("MODEL_DATA") + f"/{base_name}.internal_test.ids.txt", sep=',', index=True, header=False, columns=[])
    print("================== P80 ==================")
    print(p80['label'].value_counts())
    print("================== P20 ==================")
    print(p20['label'].value_counts())

    p80.loc[p80['label'].isin(['HD', 'XR']), 'stats_group'] = 'HD'
    p80.loc[p80['label'].isin(['AA']), 'stats_group'] = 'AA'
    p80.loc[p80['label'].isin(['CRC']), 'stats_group'] = 'CRC'


    config = {
        'zr1': [['HD', 'XR'], ['CRC']],
        'zr10': [['HD', 'XR'], ['AA', 'CRC']],
        'zr11': [['HD', 'XR'], ['AA']],
        'zr2': [['HD'], ['AA', 'CRC']],
        'zr6': [['HD'], ['CRC']],
        'zr8': [['HD'], ['AA']],
    }
    for exp_name in config:
        neg_ids = p80.loc[
            p80['label'].isin(config[exp_name][0])
        ]
        pos_ids = p80.loc[
            p80['label'].isin(config[exp_name][1]) |
            p80['stage'].isin(config[exp_name][1])
        ]
        trn_val_ids = p20.loc[
            p20['label'].isin(config[exp_name][1] + config[exp_name][0]) |
            p20['stage'].isin(config[exp_name][1])
        ]
        trn_ids = p80.loc[
            p80['label'].isin(config[exp_name][1] + config[exp_name][0]) |
            p80['stage'].isin(config[exp_name][1])
        ]
        p100.to_csv(get_location("MODEL_DATA") + f"/{base_name}_{exp_name}_{seed}.full.ids.txt", sep=',', index=True, header=False, columns=[])
        neg_ids.to_csv(get_location("MODEL_DATA") + f"/{base_name}_{exp_name}_{seed}.neg.ids.txt", sep=',', index=True, header=False, columns=[])
        pos_ids.to_csv(get_location("MODEL_DATA") + f"/{base_name}_{exp_name}_{seed}.pos.ids.txt", sep=',', index=True, header=False, columns=[])
        trn_val_ids.to_csv(get_location("MODEL_DATA") + f"/{base_name}_{exp_name}_{seed}.trn_val.ids.txt", sep=',', index=True, header=False, columns=[])
        trn_ids.to_csv(get_location("MODEL_DATA") + f"/{base_name}_{exp_name}_{seed}.trn.ids.txt", sep=',', index=True, header=False, columns=[])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('exp_name', help='实验名称 (如 zheer)')
    parser.add_argument('working_dir', help='工作目录')
    args = parser.parse_args()

    start_time = datetime.now()
    main(args)
    end_time = datetime.now()
    elapsed_time = end_time - start_time

    print(f"程序运行时间: {elapsed_time}")
