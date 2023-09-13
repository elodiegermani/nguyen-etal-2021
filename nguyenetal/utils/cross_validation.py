from scipy import stats
from sklearn import model_selection, pipeline, preprocessing, impute, feature_selection, metrics, svm, linear_model, ensemble
import copy
import pandas as pd

class StratifiedKFoldContinuous(model_selection.StratifiedKFold):
    def __init__(self, n_splits=10, n_bins=3, shuffle=True, random_state=989):
        self.n_bins = n_bins
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state = random_state)

    def split(self, X, y, groups=None):
        yBinned = pd.qcut(y, self.n_bins, labels=False, duplicates='drop')
        # yBinned.index = y.index
        return super(StratifiedKFoldContinuous, self).split(X, yBinned, groups)