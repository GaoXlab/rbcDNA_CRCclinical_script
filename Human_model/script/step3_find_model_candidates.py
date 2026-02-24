import argparse
import re

from IndexedModel import IndexedLabeledModel

def find_best_models_filter_by_correlation(df, metric, top_n, low, high):
    df_filtered = df[(df['n_pcas']>=low) & (df['n_pcas']<=high)]
    df_sorted = df_filtered.sort_values(by=metric, ascending=False)
    return df_sorted.head(top_n)

if __name__ == "__main__":
    import pandas as pd
    import glob
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str, help='zheer')
    args = parser.parse_args()
    exp_name = args.exp_name
    csv_files = glob.glob(f'search_trncv_v2_{exp_name}_*.csv')
    pattern = re.compile("zheer_zr(\d+)_\d+")

    labeled_class = IndexedLabeledModel.create_labeled_model_class(f"zheer")
    models = []
    low = 10
    for file in csv_files:
        print(f"processing {file}")
        df = pd.read_csv(file)
        discovery = pattern.findall(file)[0]
        feature = pattern.findall(file)[1]
        group_name = f"{discovery}in{feature}"
        if group_name in ['11in11', '11in8', '8in8']:
            high = 20
        else:
            high = 15
        print(f"Processing file: {file} with discovery: {discovery} and feature: {feature}")

        xgbm_model = df.copy()
        if not xgbm_model['TRNCV-AUC'].isna().any():
            crc_models = find_best_models_filter_by_correlation(xgbm_model, 'TRNCV-AUC', 10, low, high)
            crc_models['select_kind'] = 'crc'
            crc_models['group'] = group_name
            models.append(crc_models)
        if not xgbm_model['TRNCV-AUC-AA'].isna().any():

            aa_models = find_best_models_filter_by_correlation(xgbm_model, 'TRNCV-AUC-AA', 10, low, high)
            aa_models['select_kind'] = 'aa'
            aa_models['group'] = group_name

            models.append(aa_models)
        print("==============================")
    print(models)
    df_model = pd.concat(models)
    df_model.to_csv(f'selected_model_{exp_name}.csv', index=False, sep='\t')