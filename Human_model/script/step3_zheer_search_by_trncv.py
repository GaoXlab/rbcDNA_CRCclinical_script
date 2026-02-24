import os
from datetime import datetime

import argparse
import pandas as pd
from sklearn.model_selection import ParameterGrid

from ExisitsPipelinesZheer import build_exists_pipelines
from configs.params_zheer import MODEL_PARAMS
from hy.FeatureFactory import FeatureFactory
from hy.data_loader import load_sample_info, load_separate_cohorts
from hy.evaluate import generate_report, save_report
from hy.model import train_pipelines


def get_location(location):
    if location == "WORKING_DIR":
        return args.working_dir
    elif location == "MODEL_DATA":
        return os.path.join(args.working_dir, "modelData")
    elif location == "REPORT":
        return os.path.join(args.working_dir, "results/3_FeatureReduction")
    else:
        return os.path.join(args.working_dir, "results")

def main(args):
    # 加载数据
    sample_info = load_sample_info(get_location("MODEL_DATA"), 'dev')
    discovery_type = [args.dt, ]
    feature_type = [args.ft, ]
    ffs = {}
    for ft in feature_type:
        ffs[ft] = FeatureFactory(ft, 1000)
        ffs[ft].init(zr_gam_dir=args.working_dir, zr_exp_dir=args.working_dir)
    discoveries = {}
    for dt in discovery_type:
        discoveries[dt] = load_separate_cohorts(get_location("MODEL_DATA"), dt, "trn")
    if args.npca > 0:
        npca_array = [args.npca]
    else:
        npca_array = list(range(8, 20))
    params_grid = {
        'discovery': discovery_type,
        'feature': feature_type,
        'n_pcas': npca_array ,
        'top_n': range(0, 6),
        'n_skip': [0],
        'scaler': ['StandardScaler'],
        'from_pca': range(1, 7),
        'svd_solver': ['full'],
    }
    pg = ParameterGrid(params_grid)
    model_params = MODEL_PARAMS.copy()
    output_filename = f"./search_trncv_v2_{args.dt}_{args.ft}.csv"
    for current_param in pg:
        if current_param['n_pcas'] <= current_param['n_skip']:
            continue
        if current_param['n_pcas'] < current_param['from_pca']:
            continue
        if current_param['n_skip'] >= current_param['from_pca']:
            continue
        if current_param['top_n'] == 0 and current_param['from_pca'] > 1:
            continue

        discovery = discoveries[current_param['discovery']]
        feature = ffs[current_param['feature']]

        X_train = feature.fetch_feature(discovery)
        y_train = sample_info.loc[discovery.index]['target']
        model_params.update({
            'pca_params': {
                'n_pcas': current_param['n_pcas'],
                'top_n': current_param['top_n'],
                'n_skip': current_param['n_skip'],
                'scaler_name': current_param['scaler'],
                'from_pca': current_param['from_pca'],
                'svd_solver': current_param['svd_solver'],
            }
        })
        all_pipelines = build_exists_pipelines(model_params)
        pipelines = {k: all_pipelines[k] for k in ['PCABasedFeatureCombiner.xgbm',]}
        models, oof_results = train_pipelines(pipelines, X_train, y_train, model_params=model_params)

        for model_name in oof_results:
            all_result = {
                'train_cv': oof_results[model_name].loc[discovery.index],
            }
            report = generate_report(all_result, sample_info, [])
            line_report = current_param.copy()
            df = pd.json_normalize(line_report, sep='.')
            df['model_name'] = model_name
            df['TRNCV-AUC'] = report[0]['AUC']
            df['TRNCV-AUC-AA'] = report[0]['AUC-AA']
            save_report(df, output_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('working_dir', help='工作目录')
    parser.add_argument('--dt', help='discovery type')
    parser.add_argument('--ft', help='feature type')
    parser.add_argument('--npca', help='n_pca to search', type=int, default=-1)
    args = parser.parse_args()

    start_time = datetime.now()
    main(args)
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"程序运行时间: {elapsed_time}")
