import os

import numpy as np
import pandas as pd

from hy.data_loader import load_separate_cohorts, load_sample_info
from typing import Type

def merge_m1_scores(p1, p2, t1, t2):
    m = np.maximum(p1 - t1, p2 - t2)
    m_min = max(-t1, -t2)
    m_max = max(1 - t1, 1 - t2)
    s = (m - m_min) / (m_max - m_min + 1e-12)
    return s

def merge_low_high_scores(series1: pd.Series,
                 series2: pd.Series,
                 low: float,
                 high: float) -> pd.Series:
    """合并两个评分Series（简化版）"""
    mid = (low + high) / 2
    def calculate_score(val1, val2):
        if pd.isna(val2):
            return val1
        if (val1 < low) or (val1 > high):
            return val1
        return mid + (val2 - 0.5) * (high - low)
    result = []
    for index, row in series1.items():
        result.append(calculate_score(series1[index], series2[index]))
    return pd.Series(
        result,
        index=series1.index,
        name='merged_score'
    )

class IndexedModelResult(pd.Series):

    def __init__(self, data=None, **kwargs) -> None:
        if isinstance(data, pd.DataFrame):
            data = data[0]
        super().__init__(data, **kwargs)

    def set_merge_type(self, merge_type):
        self.merge_type = merge_type

    def get_merge_type(self):
        return getattr(self, 'merge_type', None)
    @property
    def _constructor(self):
        """重写构造函数，确保操作返回当前类的实例"""
        return IndexedModelResult

    @property
    def _constructor_expanddim(self):
        """重写扩展维度构造函数"""
        # 如果需要返回DataFrame时的行为
        return pd.DataFrame

    def merge_with_high_low(self, other:'IndexedModelResult', low, high):
        r = IndexedModelResult(merge_low_high_scores(self, other, low, high))
        r.set_merge_type('highlow')
        return r

    def merge_with_max(self, other, cutoff_1 = 0.5, cutoff_2 = 0.5):
        r = IndexedModelResult(merge_m1_scores(self, other, cutoff_1, cutoff_2))
        r.set_merge_type('merge')
        return r

    def merge_with_min(self, other, cutoff_1 = 0.5, cutoff_2 = 0.5):
        return IndexedModelResult()


class IndexedModel:

    CONFIG_FILE_LOCATION = "./selected_model.csv"
    CONFIG_FILE_MODEL_DATA_DIR = "./modelData/"
    DEFAULT_OOF_PRED_DIR = "./oof_preds/"
    SAMPLE_INFO = None
    def __init__(self, index, config=None):
        self.index = index
        all_configs = self.__class__.get_config()
        if index > 0:
            self.config = all_configs[all_configs['index_no'] == self.index].iloc[0]
        else:
            self.config = config

    def get_name(self):
        config = self.config
        return f"{config['discovery']}_{config['feature']}_{config['model_name']}_index_{config['index_no']}_pred.csv"

    @classmethod
    def get_config(cls):
        if not hasattr(cls, '_all_configs'):
            cls._all_configs = pd.read_csv(cls.CONFIG_FILE_LOCATION, sep="\t",
                                  dtype={'from_pca': 'Int64', 'n_pcas': 'Int64', 'n_skip': 'Int64', 'top_n': 'Int64'})
        return cls._all_configs

    @classmethod
    def get_sample_info(cls):
        if cls.SAMPLE_INFO is None:
            cls.SAMPLE_INFO = load_sample_info(cls.CONFIG_FILE_MODEL_DATA_DIR, 'dev')
        return cls.SAMPLE_INFO

    @classmethod
    def get_all_model_results(cls):
        all_configs = cls.get_config()
        results = {}
        for index, config in all_configs.iterrows():
            results[config['index_no']] = cls(config['index_no']).get_default_oof_result()
        return results

    @classmethod
    def get_all_models(cls):
        all_configs = cls.get_config()
        models = {}
        for index, config in all_configs.iterrows():
            models[config['index_no']] = cls(config['index_no'])
        return models
    @classmethod
    def get_all_groups(cls):
        all_configs = cls.get_config()
        return all_configs.groupby('group')['index_no'].apply(list).to_dict()
    def is_same_source(self, other:'IndexedModel'):
        if not pd.isna(self.config['group']) and not pd.isna(other.config['group']):
            return self.config['group'] == other.config['group']
        return self.config['discovery'] == other.config['discovery'] and self.config['feature'] == other.config['feature']

    def repeat_and_predict(self, model_params, trncv_only = False, test_ids=None):
        from ExisitsPipelinesZheer import build_exists_pipelines
        from hy.FeatureFactory import FeatureFactory
        from hy.model import train_pipelines, run_model

        type, cutoff1, cutoff2 = self.extract_combine_params()
        if type is not None:
            if test_ids is not None:
                raise ValueError("组合模型不支持传入 test_ids 参数")
            return self.repeat_combine_result()
        model_params = model_params.copy()
        sample_info = load_sample_info(self.CONFIG_FILE_MODEL_DATA_DIR, 'zr')
        config = self.config

        if pd.isna(config['from_pca']):
            return None
        model_params.update({
            'pca_params': {
                'from_pca': config['from_pca'],  # range(0, 20, 5),
                'n_pcas': config['n_pcas'],  # range(20, 100, 10),
                'n_skip': config['n_skip'],  # number of features to skip
                'scaler_name': config['scaler'],  # ['MinMaxScaler', 'StandardScaler'],
                'svd_solver': config['svd_solver'],  # ['auto', 'full'],
                'top_n': config['top_n'],  # 6	12	2	StandardScaler	auto	5
            }

        })
        current_discovery = load_separate_cohorts(self.CONFIG_FILE_MODEL_DATA_DIR, config['discovery'], "trn")

        # diff tj_cand index  and discovery index
        # message_to_feishu(f"left tj_candidate samples: {len(tj_cand)}/127")
        ff = FeatureFactory(config['feature'], int(config.get('selected_features', 1000)))
        ff.init(zr_exp_dir="hy/", zr_gam_dir="hy/")
        X_train = ff.fetch_features(current_discovery)
        all_pipelines = build_exists_pipelines(model_params)
        # pipelines = all_pipelines
        pipelines = {k: all_pipelines[k] for k in
                     [config['model_name']]}

        y_train = sample_info.loc[current_discovery.index]['target']
        model, oof_result = train_pipelines(pipelines, X_train, y_train, model_params=model_params)
        oof_result = oof_result[config['model_name']]

        if trncv_only:
            return IndexedModelResult(oof_result)
        all_pred_result = None
        if test_ids is not None:
            X_test = ff.fetch_feature(test_ids)
            all_pred_result = run_model(model, X_test)[config['model_name']]
            return IndexedModelResult(all_pred_result)

        final_result = pd.concat([oof_result, all_pred_result], axis=0)
        return IndexedModelResult(final_result)

    _OOF_CACHE = {}

    def get_default_oof_result(self):
        # 获取当前实例的名称作为 Key
        cache_key = self.get_name()

        # 2. 检查缓存是否已经存在
        if cache_key in self.__class__._OOF_CACHE:
            return self.__class__._OOF_CACHE[cache_key]

        # 原有的逻辑
        filename = f"{self.__class__.DEFAULT_OOF_PRED_DIR}/{cache_key}"
        if not os.path.isfile(filename):
            return None

        # 读取数据
        result_df = pd.read_csv(
            filename,
            skiprows=1,
            names=["seqID", 0],
            usecols=[0, 1],
            dtype={1: "float"}
        ).set_index('seqID')

        result = IndexedModelResult(result_df)

        # 3. 存入类级别缓存
        self.__class__._OOF_CACHE[cache_key] = result

        return result

    def has_default_oof_result(self):
        filename = f"{self.__class__.DEFAULT_OOF_PRED_DIR}/{self.get_name()}"
        return os.path.isfile(filename)
    def save_external_oof_result(self, result:IndexedModelResult):
        #check if dir exists
        import os
        if not os.path.exists(self.__class__.DEFAULT_OOF_PRED_DIR):
            os.makedirs(self.__class__.DEFAULT_OOF_PRED_DIR, exist_ok=True)
        result.to_csv(f"{self.__class__.DEFAULT_OOF_PRED_DIR}/{self.get_name()}", sep=',', header=True)

    def extract_zr_number(self):
        import re
        """只提取zr后面的数字部分"""
        config = self.config
        feature = config['feature']
        discovery = config['discovery']
        pattern = r'zr(\d+)'  # 使用括号捕获组
        dt_matches = re.findall(pattern, discovery)
        ft_matches = re.findall(pattern, feature)
        return int(dt_matches[0]), int(ft_matches[0])
    def extract_sub_model_indices(self):
        return self.extract_index_numbers()
    def sub_model_group_label(self):
        indices = self.extract_sub_model_indices()
        groups = []
        for idx in indices:
            sub_model = self.__class__(idx)
            groups.append(sub_model.config['group'])
        return ",".join(sorted(groups))
    def extract_index_numbers(self):
        import re
        """提取feature和discovery中的数字部分"""
        config = self.config
        discovery = config['discovery']
        feature = config['feature']

        ft_matches = []
        if not pd.isna(feature):
            pattern = r'index_(\d+)'  # 使用括号捕获组
            dt_matches = re.findall(pattern, discovery)
            ft_matches = re.findall(pattern, feature)
            return [int(x) for x in dt_matches + ft_matches]

        else:
            pattern = r'(\d+)'
            dt_matches = re.findall(pattern, discovery)
            return [int(x) for x in dt_matches]
    def expr(self):
        config = self.config
        if config['model_name'].startswith("highlow"):
            index_a, index_b = self.extract_index_numbers()
            return f"({self.__class__(index_a).expr()}+{self.__class__(index_b).expr()})"
        if config['model_name'].startswith("merge"):
            index_a, index_b = self.extract_index_numbers()
            return f"({self.__class__(index_a).expr()}|{self.__class__(index_b).expr()})"
        if config['model_name'].startswith("vote"):
            expression = self.config['discovery']
            index_a, index_b, index_c = map(int, expression.split('+'))
            return f"({self.__class__(index_a).expr()},{self.__class__(index_b).expr()},{self.__class__(index_c).expr()})"
        zr_dt, zr_ft = self.extract_zr_number()
        return f"{zr_dt}in{zr_ft}"

    def get_sub_models(self):
        return [self.__class__(x) for x in self.extract_index_numbers()]

    def get_all_sub_models(self):
        if pd.isna(self.config['hash']):
            return []
        print(self.config['hash'], self.config['group'])
        sub_models =self.get_sub_models()
        all_sub_models = sub_models.copy()
        for sm in sub_models:
            all_sub_models.extend(sm.get_all_sub_models())
        return all_sub_models

    def extract_combine_params(self):
        if "-" not in self.config['model_name']:
            return None, None, None
        merge_type, str_cutoff1, str_cutoff2 = self.config['model_name'].split("-", 2)
        if merge_type not in ['highlow', 'merge']:
            return merge_type, str_cutoff1, str_cutoff2
        return merge_type, float(str_cutoff1), float(str_cutoff2)

    def merge_with_params(self, result_a, result_b, param):
        type, cutoff1, cutoff2 = self.extract_combine_params()
        if type == 'highlow':
            return result_a.merge_with_high_low(result_b, param['low'], param['high'])
        elif type == 'merge':
            return result_a.merge_with_max(result_b, param['cutoff_1'], param['cutoff_2'])

    @classmethod
    def load_temp_result(cls, merge_type, result_hash):
        dir = f"{cls.DEFAULT_OOF_PRED_DIR}/search_results_pred/{merge_type}/"
        return IndexedModelResult(pd.read_csv(f"{dir}/pred_{result_hash}.csv", skiprows=1, names=["seqID", 0], usecols=[0, 1], dtype={1: "float"}).set_index('seqID'))

    def repeat_combine_result(self):
        config = self.config
        if config['model_name'].startswith("vote"):
            expression = self.config['discovery']
            index_ids = map(int, expression.split('+'))
            sub_results = []
            for index_id in index_ids:
                sub_results.append(self.__class__(index_id).get_default_oof_result())
            return IndexedModelResult(pd.concat(sub_results, axis=1).mean(axis=1))
        sub_models = self.get_sub_models()
        r1 = sub_models[0].get_default_oof_result()
        r2 = sub_models[1].get_default_oof_result()
        _, cutoff1, cutoff2 = self.extract_combine_params()
        return self.merge_with_params(r1, r2, {
            'low': cutoff1,
            'high': cutoff2,
            'cutoff_1': cutoff1,
            'cutoff_2': cutoff2,
        })

class IndexedTrainCvModel(IndexedModel):

    DEFAULT_OOF_PRED_DIR = "./preds_trncv"
    CONFIG_FILE_LOCATION = "./selected_model_trncv.csv"
    DEFAULT_REPORTS_DIR = "./reports/"
    round_name = ""
    def get_name(self):
        config = self.config
        if config['model_name'].startswith("highlow"):
            return f"pred_{config['index_no']}.{config['discovery']}+{config['feature']}_{config['model_name']}.csv"
        if config['model_name'].startswith("merge"):
            return f"pred_{config['index_no']}.{config['discovery']}|{config['feature']}_{config['model_name']}.csv"
        return f"pred_{config['index_no']}.{config['discovery']}_{config['feature']}_{config['model_name']}.csv"

    @classmethod
    def get_batch_results(cls, batch_name):
        import glob
        report_csv_path = f"./reports/trn_cv_{cls.round_name}/{batch_name}/"
        print(f"{report_csv_path}/pair*.csv")
        results_files = glob.glob(f"{report_csv_path}/pair*.csv")
        return pd.concat((pd.read_csv(f, on_bad_lines='warn') for f in results_files), ignore_index=True)

    @classmethod
    def get_mode_by_batch(cls, batch_name):
        csvs_dir = cls.DEFAULT_REPORTS_DIR + f"/{batch_name}/"
        csv_fils = [f for f in os.listdir(csvs_dir) if f.endswith('.csv')]
        parts = csv_fils[0].split('_')
        mode = 'unknown'
        if len(parts) >= 4:
            mode = parts[-2]  # 取倒数第二个部分
        return mode

    @classmethod
    def from_report_line(cls, mode, report_line):
        report_line = report_line.copy()
        if mode == 'voter':
            report_line['discovery'] = report_line['pair']
            report_line['model_name'] = f"voter-{report_line['pair']}-"
        elif mode == 'merge' or mode == 'highlow':
            try:
                discovery, feature = report_line['pair'].split('+')
            except ValueError:
                return
            report_line['discovery'] = discovery
            report_line['feature'] = feature
            report_line['model_name'] = f"{mode}-{report_line['low_param']}-{report_line['high_param']}"
        return cls(-1, config=report_line)

class IndexedLabeledModel:
    @classmethod
    def create_labeled_model_class(cls, round_name)-> Type[IndexedTrainCvModel]:
        return type(f'IndexedLabeledModel{round_name}', (IndexedTrainCvModel,), {
            'DEFAULT_OOF_PRED_DIR': f"./preds_trncv_{round_name}",
            'DEFAULT_REPORTS_DIR': f"./reports/trn_cv_{round_name}",
            'CONFIG_FILE_LOCATION': f"./selected_model_trncv_{round_name}.csv",
            'round_name': round_name,
        })
