import argparse
import os
import numpy as np
from datetime import datetime

import pandas as pd


def main(args):
    base_name = args.exp_name
    modelData_dir = args.modelData_dir

    excel_file_name = os.path.join(modelData_dir, "Participant_characteristics.xlsx")
    clin_data = pd.read_excel(excel_file_name)

    sampleinfo_data = clin_data.copy()

    #translate stage and label

    def translate_label_and_stage(df):
        hd_sub = ['Negative colonoscopy', 'Nonneoplastic findings']
        label_conditions = [
            (df['Group'].isin(['Non-AN control', 'Control'])) & (df['Sub-group'].isin(hd_sub)),
            (df['Group'].isin(['Non-AN control', 'Control'])) & (~df['Sub-group'].isin(hd_sub)),
            (df['Group'] == 'AA'),
            (df['Group'] == 'CRC')
        ]
        label_choices = ['HD', 'XR', 'AA', 'CRC']

        df['label'] = np.select(label_conditions, label_choices, default=df['Group'])
        df['stage'] = np.where(df['label'] == 'CRC', df['Stage'], df['label'])
        df['target'] = np.where(df['Group'].isin(['AA', 'CRC']), 1, 0)
        df.rename(columns={'Sample': 'seqID'}, inplace=True) #向前兼容
        return df

    sampleinfo_data = translate_label_and_stage(sampleinfo_data)
    output_dict = {
        'dev': ['Discovery', 'Internal test'],
        'ind_wz': 'WENZHOU',
        'ind_sd': 'SHANDONG',

    }

    def split_df_by_cohort(df, cohort_dict):
        result_dfs = {}
        for key, value in cohort_dict.items():
            if isinstance(value, list):
                result_dfs[key] = df[df['Cohort'].isin(value)]
            else:
                result_dfs[key] = df[df['Cohort'] == value]
        return result_dfs

    split_dfs = split_df_by_cohort(sampleinfo_data, output_dict)

    for key, info in split_dfs.items():
        info = info[['seqID', 'target', 'stage', 'label',]]
        info.to_csv(os.path.join(modelData_dir, f"sampleinfo.{key}.txt"), index=False, sep="\t")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('exp_name', help='实验名称 (如 zheer)')
    parser.add_argument('modelData_dir', help='工作目录')
    args = parser.parse_args()

    start_time = datetime.now()
    main(args)
    end_time = datetime.now()
    elapsed_time = end_time - start_time

    print(f"程序运行时间: {elapsed_time}")
