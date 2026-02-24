import os
from .Estimator import BedFeatureSelector
from .data_loader import load_normalized_tab_file

class FeatureFactory:
    """
    A factory class for creating feature objects.
    """

    def __init__(self, feature_type, selected_by = None):
        self.feature_type = feature_type
        self.selected_by = selected_by
        self._features = None
        self._normalized_data = None
        self._order_bed_file = None

    def init(self, zr_gam_dir = None, zr_exp_dir = None):
        environment = os.environ.get('ENVIRONMENT', 'DEV')
        if zr_gam_dir is None:
            raise ValueError(f"Unknown environment: {environment}")
        if zr_exp_dir is None:
            raise ValueError(f"Unknown environment: {environment}")

        filename = f"{zr_gam_dir}/normalized_results/{self.feature_type}/train_gam.tab.{self.feature_type}"
        normalized_data = load_normalized_tab_file(filename)

        self._normalized_data = normalized_data
        self._order_bed_file = f'{zr_exp_dir}/{self.feature_type}/all.{self.feature_type}.bed.out'
        self.reselect_features()

    def fetch_feature(self, ids):
        """
        Create a feature object based on the feature type.

        :param feature_type: The type of the feature to create.
        :param args: Positional arguments for the feature constructor.
        :param kwargs: Keyword arguments for the feature constructor.
        :return: An instance of the specified feature type.
        """
        if not self.inited():
            self.init()

        return self._features.loc[ids.index]

    def inited(self):
        return self._features is not None

    def reselect_features(self, selected_by=None):
        # if selected_by is integer
        if self._normalized_data is None:
            raise ValueError("FeatureFactory is not initialized. Please call init() first.")
        if selected_by is not None:
            self.selected_by = selected_by
        self._features = None
        if self.selected_by is None:
            self._features = self._normalized_data
            return
        if isinstance(self.selected_by, int):
            top_n_selector = BedFeatureSelector(
                bed_path=self._order_bed_file,
                top_n=self.selected_by,
            )
            self._features = top_n_selector.fit_transform(self._normalized_data)
        else:
            # if selected_by is file and file exists
            if not os.path.exists(self.selected_by):
                raise ValueError(f"File {self.selected_by} does not exist")

            top_n_selector = BedFeatureSelector(
                bed_path=self.selected_by,
                top_n=-1,
            )
            self._features = top_n_selector.fit_transform(self._normalized_data)

    def fetch_features(self, *ids):
        """批量获取多个id对应的特征

        Args:
            *ids: 可变数量的id参数

        Returns:
            list: 按输入顺序返回各id对应的特征
        """
        return [self.fetch_feature(id) for id in ids]

