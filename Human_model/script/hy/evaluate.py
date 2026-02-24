import hashlib
import math
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def calculate_thresholds(y_true: np.ndarray, y_proba, specificity_targets = None) -> dict:
    """根据交叉验证结果计算不同特异性要求的阈值

    Args:
        y_true: 真实标签 (ground truth)
        y_proba: 预测概率 (正类概率)
        specificity_targets: 目标特异性百分比列表 (0-100)

    Returns:
        dict: 键为特异性值，值为对应阈值及敏感性 {spec%: {'threshold': ..., 'sensitivity': ...}}
    """
    if isinstance(y_proba, pd.DataFrame):
        y_proba = y_proba[y_proba.columns[0]]
    if specificity_targets is None:
        specificity_targets = [90, 95, 98]
    prob_df = pd.DataFrame({
        'y_true': y_true,
        'y_proba': y_proba,
    }).sort_values(by='y_proba').reset_index(drop=True)
    prob_hd_df = prob_df[prob_df['y_true'] == 0].copy().reset_index()
    total_hd = len(prob_df[prob_df['y_true'] == 0])
    thresholds_dict = {}
    for target in specificity_targets:
        target_hd_index = math.floor(total_hd * (target / 100.0) + 1 + 1e-8)
        cutoff = prob_hd_df.loc[target_hd_index - 1, 'y_proba']
        min_cutoff = prob_df[prob_df['y_proba'] > cutoff]['y_proba'].min()
        thresholds_dict[f"{target}%"] = {
            'threshold': float(min_cutoff)
        }
    return thresholds_dict

_file_status_cache = defaultdict(bool)
def save_report(metrics, filename, lock_type='global'):
    """
    保存指标到文本文件

    Args:
        metrics: 要保存的指标数据
        filename: 目标文件名
        lock_type: 锁类型，'local' 或 'global'
            - 'local': 使用进程内线程锁，性能更高
            - 'global': 使用文件锁，支持跨进程
    """
    from filelock import FileLock
    from pandas import json_normalize
    if lock_type == 'local':
        lock_file = f"/dev/shm/{hashlib.md5(filename.encode()).hexdigest()}.lock"
    elif lock_type == 'global':
        lock_file = filename + '.lock'
    else:
        raise ValueError("lock_type 必须是 'local' 或 'global'")
    lock = FileLock(lock_file)
    if not isinstance(metrics, pd.DataFrame):
        metrics = json_normalize(metrics, max_level=3)
    with lock:
        if _file_status_cache[filename] or os.path.isfile(filename):
            metrics.to_csv(filename, mode='a', index=False, header=False)
        else:
            metrics.to_csv(filename, index=False)
            time.sleep(1)
        _file_status_cache[filename] = True

from sklearn.metrics import roc_curve


def generate_zheer_auc_report(zr_all_p20_result, info):
    # 1. 数据标准化：确保结果是 DataFrame 且列名统一
    if isinstance(zr_all_p20_result, pd.Series):
        scores = zr_all_p20_result.to_frame('score')
    else:
        scores = zr_all_p20_result.rename(columns={zr_all_p20_result.columns[0]: 'score'})

    # 2. 预对齐：一次性完成索引匹配，减少后续 .loc 开销
    common_ids = scores.index.intersection(info.index)
    working_df = info.loc[common_ids, ['label', 'target']].copy()
    working_df['score'] = scores.loc[common_ids, 'score']

    # 3. 定义对比组配置 (组名: 包含的标签列表)
    groups_config = {
        'CRC_HD_XR': ['CRC', 'HD', 'XR'],
        'AA_HD_XR': ['AA', 'HD', 'XR'],
        'ALL_HD_XR': ['CRC', 'AA', 'HD', 'XR'],
    }

    # 4. 循环计算结果
    report = {}
    for name, labels in groups_config.items():
        # 使用 query 或 boolean mask 过滤数据
        subset = working_df[working_df['label'].isin(labels)]

        if not subset.empty:
            auc = roc_auc_score(subset['target'], subset['score'])
            report[name] = round(auc, 6)
        else:
            report[name] = None  # 或者 0.0，视业务需求而定

    return report

def calculate_auc(y_true, y_scores):
    """计算AUC值

    Args:
        y_true: 真实标签 (ground truth)
        y_scores: 预测分数 (scores)

    Returns:
        float: AUC值
    """
    return roc_auc_score(y_true, y_scores)
def generate_report(t, info, cutoffs, plot_auc=False, save_path=None):
    results = []


    for t_type in t:
        pred = t[t_type]
        if isinstance(pred, pd.Series):
            pred = pred.to_frame(0)
        data = pred.join(info[['target', 'stage']], lsuffix='_left')
        all_data = data.copy()
        aa_data = data[~data['stage'].isin(['0', 'I', 'II', 'III', 'IV'])]
        data = data[data['stage'] != 'AA']
        # if any target = 1 in aa_data, then set include_aa = True
        include_aa = aa_data['target'].sum() > 0
        if include_aa:
            aa_roc_auc = round(roc_auc_score(aa_data['target'], aa_data[0]), 6)
        else:
            aa_roc_auc = 'N/A'

        # fill all nan target with 0
        data['target'] = data['target'].fillna(0)

        # Check if we have both positive and negative samples
        unique_targets = data['target'].unique()
        has_both_classes = len(unique_targets) > 1

        line = {'type': t_type}

        # Only calculate ROC and AUC if we have both classes
        if has_both_classes:
            fpr, tpr, _ = roc_curve(data['target'], data[0])
            try:
                roc_auc = round(roc_auc_score(data['target'], data[0]), 6)
            except ValueError:
                roc_auc = 0.0
            all_roc_auc = round(roc_auc_score(all_data['target'], all_data[0]), 6)
            line['AUC'] = roc_auc
            line['AUC-ALL'] = all_roc_auc

        else:
            line['AUC'] = 'N/A'
            line['AUC-ALL'] = 'N/A'

        line['AUC-AA'] = aa_roc_auc

        if len(cutoffs) == 0:
            cutoffs = {'0.5': {'threshold': 0.5}}

        for spec in cutoffs:
            threshold = cutoffs[spec]['threshold']
            line[spec] = {}

            # Calculate sensitivity only if we have positive samples
            if 1 in unique_targets:
                pos_samples = data[data['target'] == 1]
                line[spec]['sens'] = round(pos_samples.apply(
                    lambda row: 1 if row[0] >= threshold else 0, axis=1).sum() / len(pos_samples), 3)

                if include_aa:
                    line[spec]['sens-AA'] = round(aa_data.apply(
                        lambda row: 1 if row['target'] == 1 and row[0] >= threshold else 0, axis=1).sum() /
                        aa_data['target'].sum(), 3)
                else:
                    line[spec]['sens-AA'] = 'N/A'
            if 0 in unique_targets:
                neg_samples = data[data['target'] == 0]
                line[spec]['spec'] = round(neg_samples.apply(
                    lambda row: 1 if row[0] < threshold else 0, axis=1).sum() / len(neg_samples), 3)
            else:
                line[spec]['spec'] = 'N/A'

        results.append(line)
    return results

def save_prediction(all_results, filename):
    # 为每个DataFrame添加对应的key列
    dfs = []
    for key, df in all_results.items():
        if isinstance(df, pd.Series):
            df = df.to_frame(0)
        df = df.copy()  # 避免修改原始DataFrame
        df['source_key'] = key  # 添加新列存储key
        dfs.append(df)

    # 合并所有DataFrame并保存
    pd.concat(dfs, axis=0).to_csv(filename, index=True, header=True, float_format="%.8f")