from scipy import stats
from sklearn import model_selection, pipeline, preprocessing, impute, feature_selection, metrics, svm, linear_model, ensemble
import copy
import pandas as pd

class StratifiedKFoldContinuous(model_selection.StratifiedKFold):
    '''
    Creates KFold stratified with continuous variables. 
    '''
    def __init__(self, n_splits=10, n_bins=3, shuffle=True, random_state=989):
        '''
        Parameters
        ----------
            n_splits: int
                n. of folds
            n_bins: int
                n. of bins
            shuffle: bool
                if shuffle participants
            random_state: int
                seed for randomisation
        '''
        self.n_bins = n_bins
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state = random_state)

    def split(self, X, y, groups=None):
        '''
        Split folds depending on continuous variable.

        Parameters
        ----------
            X: list
                list of subjects
            y: list
                outcome
            labels: bool
                

        '''
        yBinned = pd.qcut(y, self.n_bins, labels=False, duplicates='drop')
        # yBinned.index = y.index
        return super(StratifiedKFoldContinuous, self).split(X, yBinned, groups)