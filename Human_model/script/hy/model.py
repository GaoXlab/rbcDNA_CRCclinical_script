import os

import joblib
import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, KFold

set_config(transform_output="pandas")


def run_model(models, X_test ):
    y_probas = pd.DataFrame()
    for model_type in models:
        model = models[model_type]
        #
        X_test_aligned = X_test[model.fitted_features_]
        #
        if model_type != 'nn':
            y_proba = model.predict_proba(X_test_aligned)[:, 1]
        else:
            # 使用模型预测
            y_proba = model.predict(X_test_aligned)
            # 处理预测结果
            y_proba = y_proba.flatten()
        y_probas[model_type] = pd.Series(y_proba, index=X_test.index)

    return y_probas

def train_pipelines(pipelines, X, y: pd.Series, cv_splits=5, model_params=None, is_regression=False, oof_test=None):
    oof_predictions = {}
    models = {}
    for mt in pipelines:
        print(mt)
        # 检查是否是 pipeline builder（即是否有 build 方法）
        if hasattr(pipelines[mt], 'build'):
            pipe = pipelines[mt].build()  # 只有 pipeline builder 才调用 build()
        else:
            pipe = pipelines[mt]  # 直接使用已有的 pipeline
        pipe, oof_proba = run_cv_pipeline_ext(pipe, X, y, cv_splits, model_params, is_regression=is_regression, oof_test=oof_test)
        models[mt] = pipe
        oof_predictions[mt] = oof_proba
    return models, oof_predictions

def train_pipeline(pipeline, X, y: pd.Series, cv_splits=5, model_params=None, oof_test=None):

    if hasattr(pipeline, 'build'):
        pipe = pipeline.build()  # 只有 pipeline builder 才调用 build()
    else:
        pipe = pipeline

    pipe, oof_proba = run_cv_pipeline_ext(pipe, X, y, cv_splits, model_params, False, oof_test)
    return pipe, oof_proba

def run_cv_pipeline(pipeline, X: pd.DataFrame, y: pd.Series, cv_splits=5, model_params=None, is_regression=False):
    if model_params is None:
        model_params = {}
    oof_predictions = {}
    if is_regression:
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=model_params.get('random_state', 1234))
    else:
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=model_params['random_state'])
    pipe = pipeline
    # 交叉验证预测
    if is_regression:
        oof_proba = cross_val_predict(
            pipe, X, y, cv=cv, method='predict', n_jobs=5
        )
    else:
        oof_proba = cross_val_predict(
            pipe, X, y, cv=cv, method='predict_proba', n_jobs=5
        )[:, 1]
    oof_predictions = pd.DataFrame(oof_proba, index=X.index)

    pipe.fit(X, y)
    pipe.fitted_features_ = X.columns.tolist()

    return pipe, oof_predictions


def run_cv_pipeline_ext(pipeline, X: pd.DataFrame, y: pd.Series, cv_splits=5, model_params=None, is_regression=False, oof_test=None):
    if model_params is None:
        model_params = {}
    
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=model_params['random_state'])
    pipe = pipeline
    
    # 存储训练集的OOF预测
    oof_proba = np.zeros(len(X))
    test_predictions = []
    fold_models = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        # 训练当前fold的模型
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        fold_pipe = clone(pipe)
        fold_pipe.fit(X_train, y_train)
        fold_models.append(fold_pipe)
        
        # OOF预测
        if is_regression:
            oof_proba[val_idx] = fold_pipe.predict(X_val)[:, 1]
        else:
            oof_proba[val_idx] = fold_pipe.predict_proba(X_val)[:, 1]
        
        # 测试集预测
        if oof_test is not None and len(oof_test) > 0:
            if is_regression:
                test_pred = fold_pipe.predict(oof_test)[:, 1]
            else:
                test_pred = fold_pipe.predict_proba(oof_test)[:, 1]
            test_predictions.append(test_pred)
    
    # 创建基础结果DataFrame
    oof_predictions = pd.DataFrame(oof_proba, index=X.index, columns=[0])
    
    # 添加测试集预测
    if oof_test is not None and test_predictions:
        # 计算测试集预测的平均值
        test_pred_mean = np.mean(test_predictions, axis=0)
        test_pred_df = pd.DataFrame(test_pred_mean, index=oof_test.index, columns=[0])
        
        # 合并结果
        final_predictions = pd.concat([oof_predictions, test_pred_df], axis=0)
    else:
        final_predictions = oof_predictions
    
    pipe.fit(X, y)
    pipe.fitted_features_ = X.columns.tolist()
    pipe.fold_models_ = fold_models
    
    return pipe, final_predictions

def save_model(models, cutoffs, location, name):
    results = {
        "models": models,
        "cutoffs": cutoffs
    }
    joblib.dump(results, os.path.join(location, f"model.{name}.pkl"))
    return None

def load_model(location, name):
    saved_model = joblib.load(os.path.join(location, f"model.{name}.pkl"))
    return saved_model['models'], saved_model['cutoffs']

