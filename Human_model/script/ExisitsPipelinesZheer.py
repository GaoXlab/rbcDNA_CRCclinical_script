from hy.PipelineBuilder import PipelineBuilder
def build_exists_pipelines(model_params):
    existsPipelines = {}

    existsPipelines['PCABasedFeatureCombiner.xgbm'] = (
        PipelineBuilder(model_params)
        .add_pca_feature_combiner()
        .add_xgb_classifier()
    )

    return existsPipelines
