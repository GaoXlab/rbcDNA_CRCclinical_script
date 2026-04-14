import argparse
import pandas as pd

from configs.params_zheer import MODEL_PARAMS
from hy.IndexedModel import IndexedLabeledModel
from hy.data_loader import load_separate_cohorts, load_sample_info
from hy.evaluate import generate_zheer_auc_report

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", help="实验名称 (如 zheer)", type=str, default="zheer")
    parser.add_argument("working_dir", help="工作目录", type=str, default=".")

    args = parser.parse_args()
    exp_name = args.exp_name
    working_dir = args.working_dir

    test_ids = load_separate_cohorts('modelData', exp_name, 'internal_test')
    sampleinfo = load_sample_info('modelData', 'dev')
    labeled_class = IndexedLabeledModel.create_labeled_model_class(exp_name)
    p20_info = test_ids.join(sampleinfo, how='inner')
    models = labeled_class.get_config()
    for idx, m in models.iterrows():
        if pd.isna(m['P20-AUC.CRC_HD_XR']):

            m = labeled_class(m['index_no'])
            p20_result = m.run_model(MODEL_PARAMS, test_ids)
            print(p20_result)
            report = generate_zheer_auc_report(p20_result, sampleinfo)
            print(report)
            models.loc[idx, 'P20-AUC.CRC_HD_XR'] = report['CRC_HD_XR']
            models.loc[idx, 'P20-AUC.AA_HD_XR'] = report['AA_HD_XR']
            models.loc[idx, 'P20-AUC.ALL_HD_XR'] = report['ALL_HD_XR']
            # exit(0)

    models.to_csv(f'selected_model_{exp_name}_with_internal_test.csv', index=False, sep="\t")