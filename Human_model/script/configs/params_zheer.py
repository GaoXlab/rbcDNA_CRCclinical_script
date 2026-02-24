# 模型参数
MODEL_PARAMS = {
    'random_state': 1234,
    'pca_params': {
        'n_pcas': 20,
        'n_feas': 0,
        'n_skip': 0,
        'from_pca': 1,
    },
    'xgb_params': {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.05,
        'n_estimators': 190,
        'max_depth': 3,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 1,
        'gamma': 0.7,
        'reg_alpha': 1,
        'reg_lambda': 1,
        'random_state': 1234,
        'n_jobs': 8
    },
}