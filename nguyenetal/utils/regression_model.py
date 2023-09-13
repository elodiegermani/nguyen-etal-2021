from scipy import stats
from sklearn import model_selection, pipeline, preprocessing, impute, feature_selection, metrics, svm, linear_model, \
    ensemble
import copy
from nguyenetal.utils import cross_validation
import numpy as np
import pandas as pd

def rmse(true, predict):
    return np.sqrt(np.mean(np.square(true - predict)))

def rsquare(true, predict):
    ssTot = np.sum(np.square(true - np.mean(true)))
    ssRes = np.sum(np.square(true - predict))
    return 1 - (ssRes / (ssTot + np.finfo(float).eps))

class RegressorPanel:
    outer: model_selection.BaseCrossValidator
    inner: model_selection.BaseCrossValidator

    def __init__(self, data, target,
                 outer=3,
                 inner=5,
                 metric_rs='rsquare',
                 random_seed=432):
        """
        Set of shallow learning models for regression.

        :param data: dataframe of input data, or a path to a file containing the the input data (CSV or pkl)
        :type data: pandas.DataFrame
        :param target:
        :type target: pandas.DataFrame
        :param outer:
        :type outer:
        :param inner:
        :type inner:
        :param metric_rs: 'rsquare' or 'rmse'; which metric to use to select the best model in each random search
        :type metric_rs: str
        :param random_seed:
        :type random_seed:
        """

        self.data = data
        self.target = target

        if isinstance(outer, int):
            self.outer = model_selection.KFold(n_splits=outer, shuffle=True, random_state=random_seed)
        else:
            self.outer = outer
        if isinstance(inner, int):
            self.inner = model_selection.KFold(n_splits=inner, shuffle=True, random_state=random_seed)
        else:
            self.inner = inner

        self.model_dict = None
        self.set_default_models()
        self.preprocessing = None
        self.set_default_preprocessing()
        self.feature_selection = None
        self.metric_rs = metric_rs
        self.random_seed = random_seed

    def set_models(self, *args):
        """
        Select scikit-learn models to train. Pass in any number of tuples, where each tuple contains a scikit-learn
        regression model class and the dictionary of hyperparameter ranges to search.
        For example, set_models((sklearn.linear_model.Lasso, {'alpha': scipy.stats.uniform(0.1, 10.0),
                                                            'max_iter': scipy.stats.randint(500, 5000)},
                                sklearn.ensemble.RandomForestRegressor, {'n_estimators': stats.randint(200, 12000),
                                                                      'min_samples_split': stats.uniform(0.01, 0.5),
                                                                      'min_samples_leaf': stats.randint(1, 6),
                                                                      'max_depth': stats.randint(1, 10)}
                                )
        If models are not specified, the default models and hyperparameters will be used.

        :param args: tuples containing (regression model, hyperparameter dictionary) pairs
        :type args: tuple
        :return:
        :rtype:
        """
        self.model_dict = {model_tupple[0].__name__: model_tupple for model_tupple in args}

    def set_default_models(self):
        self.model_dict = {'ElasticNet': (linear_model.ElasticNet(max_iter=5000),
                                            {'alpha': np.logspace(0, 1.5, 1000),
                                             'l1_ratio': stats.uniform(0, 1.0)}
                                            ),
                             'LinearSVR': (svm.LinearSVR(tol=0.001, max_iter=50000),
                                           {'C': np.logspace(-3, 0, 1000, base=10),
                                            'epsilon': np.logspace(-2, 0, 1000),
                                            'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']}
                                           ),
                             'GradientBoostingRegressor': (ensemble.GradientBoostingRegressor(loss='squared_error',
                                                                                              criterion='friedman_mse'),
                                                           {'learning_rate': np.logspace(-2, -1, 1000),
                                                            'n_estimators': np.linspace(10, 1000, 1000,
                                                                                        dtype=int),
                                                            'min_samples_split': stats.uniform(0.1, 0.7),
                                                            'min_samples_leaf': stats.randint(1, 6),
                                                            'max_depth': stats.randint(1, 4)}
                                                           ),
                             'RandomForestRegressor': (ensemble.RandomForestRegressor(criterion='squared_error', random_state=432),
                                                        {'n_estimators': np.logspace(1, 3, 100).astype(int),
                                                        'min_samples_split': stats.uniform(0.01, 0.5),
                                                        'min_samples_leaf': stats.randint(1, 6),
                                                        'max_depth': stats.randint(1, 8)}
                                                       )
                             }

    def set_preprocessing(self, pipeline):
        """
        Set the input data preprocessing  pipeline that will be applied to all data before it is passed into a
        regression model. Typically, this pipeline would contain some kind of scaling/normalization and NaN imputer.
        If no pipeline is specified, the default will be used:
            pipeline.Pipeline([('scaler', preprocessing.StandardScaler()),
                                  ('imputer', impute.SimpleImputer()),
                                  ('selector', feature_selection.SelectKBest(feature_selection.f_regression))])

        :param pipeline: a scikit-learn pipeline object.
        :type pipeline: sklearn.pipeline.Pipeline
        :return:
        :rtype:
        """
        self.preprocessing = pipeline

    def set_default_preprocessing(self):
        self.preprocessing = pipeline.Pipeline([('StandardScaler', preprocessing.StandardScaler()),
                                                ('SimpleImputer', impute.SimpleImputer())])

    def set_feature_selection(self, feature_selection_tupple):
        """
        Add a feature selection method that will be used after preprocessing and before model fitting. Takes a tuple
        containing (sklearn feature selection class, dictionary of hyperparams for this selector)

        :param feature_selection_tupple: tuple containing (feature selection class, dictionary of hyperparams)
        :type feature_selection_tupple: tuple
        :return:
        :rtype:
        """
        self.feature_selection = feature_selection_tupple

    def run_single_model(self, model_name, n_iters=100, n_jobs=1):
        """
        Run a random search on one model

        :param model_name: key referring to one of the models in self.model_dict
        :type model_name: str
        :param n_iters: number of iterations in the random search
        :type n_iters: int
        :param n_jobs: number of parallel jobs
        :type n_jobs: int
        :return: result_dict: dictionary containing mean train, val, and test RMSE and Rsquare; cv_score_dict:
        dictionary containing the exhaustive results returned by sklearn.model_selection.cross_validate
        :rtype: dict, dict
        """
        if model_name not in self.model_dict.keys():
            raise ValueError('Incorrect model name: {}'.format(model_name)) # Test for name in model dictionnary

        pipe = copy.deepcopy(self.preprocessing)

        if self.feature_selection is not None: # If feature selection ? 
            selector_name = self.feature_selection[0].__class__.__name__
            pipe.steps.append((selector_name, self.feature_selection[0]))

        pipe.steps.append((model_name, self.model_dict[model_name][0]))

        param_dict_orig = self.model_dict[model_name][1]
        # Append the model name to the hyperparam keys so that RandomSearch can recognize them
        param_dict = {model_name + '__' + key: val for key, val in param_dict_orig.items()}

        if self.feature_selection is not None:
            select_orig_param_dict = self.feature_selection[1]
            select_param_dict = {selector_name + '__' + key: val for key, val in select_orig_param_dict.items()}
            param_dict = dict(param_dict, **select_param_dict)

        score_dict = {'rmse': metrics.make_scorer(rmse,
                                                   greater_is_better=False),
                       'rsquare': metrics.make_scorer(rsquare,
                                                      greater_is_better=True)}

        random = model_selection.RandomizedSearchCV(pipe, param_dict,
                                                    scoring=score_dict,
                                                    cv=self.inner,
                                                    n_iter=n_iters,
                                                    return_train_score=True,
                                                    n_jobs=n_jobs,
                                                    random_state=self.random_seed,
                                                    refit=self.metric_rs)

        cv_score_dict = model_selection.cross_validate(random, X=self.data, y=self.target, cv=self.outer,
                                                      groups=None,
                                                      scoring=score_dict,
                                                      return_train_score=True, return_estimator=True)
        if hasattr(self.outer, 'n_splits'):
            n_outer_splits = self.outer.n_splits
        elif hasattr(self.outer, 'get_n_splits'):
            n_outer_splits = self.outer.get_n_splits(self.data)
        elif hasattr(self.outer, '__len__'):
            n_outer_splits = self.outer.__len__()
            
        rmse_train = -cv_score_dict['train_rmse']
        rsquare_train = cv_score_dict['train_rsquare']
        best_model_idx = [cv_score_dict['estimator'][n_fold].best_index_ for n_fold in range(n_outer_splits)]
        
        rmse_valid = [-cv_score_dict['estimator'][n_fold].cv_results_['mean_test_rmse'][best_model_idx[n_fold]] for n_fold in
                     range(n_outer_splits)]
        rsquare_valid = [cv_score_dict['estimator'][n_fold].cv_results_['mean_test_rsquare'][best_model_idx[n_fold]] for
                        n_fold in range(n_outer_splits)]
        
        rmse_test = -cv_score_dict['test_rmse']
        rsquare_test = cv_score_dict['test_rsquare']
        
        result_dict = {'mean_train_rmse': np.mean(rmse_train),
                       'std_train_rmse': np.std(rmse_train),
                       'mean_val_rmse': np.mean(rmse_valid),
                       'std_val_rmse': np.std(rmse_valid),
                       'mean_test_rmse': np.mean(rmse_test),
                       'std_test_rmse': np.std(rmse_test),
                       'mean_train_rsquare': np.mean(rsquare_train),
                       'std_train_rsquare': np.std(rsquare_train),
                       'mean_val_rsquare': np.mean(rsquare_valid),
                       'std_val_rsquare': np.std(rsquare_valid),
                       'mean_test_rsquare': np.mean(rsquare_test),
                       'std_test_rsquare': np.std(rsquare_test)
                       }

        return result_dict, cv_score_dict

    def run_all_models(self, n_iters=100, verbose=True, n_jobs=1):
        """
        Run random hyperparam searches on all models

        :param n_iters: number of random search iterations
        :type n_iters: int
        :param verbose: flag for verbose output
        :type verbose: bool
        :param n_jobs: number of parallel jobs
        :type n_jobs: int
        :return: result dataframe
        :rtype: pandas.DataFrame
        """

        if verbose:
            #print('Writing results to {}'.format(self.strOutputDir))
            if hasattr(self.outer, 'n_splits'):
                n_outer_splits = self.outer.n_splits
            elif hasattr(self.outer, 'get_n_splits'):
                n_outer_splits = self.outer.get_n_splits(self.data)
            elif hasattr(self.outer, '__len__'):
                n_outer_splits = self.outer.__len__()
            else:
                n_outer_splits = '?'

            if hasattr(self.inner, 'n_splits'):
                n_inner_splits = self.inner.n_splits
            elif hasattr(self.inner, '__len__'):
                n_inner_splits = self.inner.__len__()
            else:
                n_inner_splits = '?'
                
            print('Using {} outer folds and {} inner folds'.format(n_outer_splits, n_inner_splits))
            print('{} subjects, {} features'.format(self.data.shape[0], self.data.shape[1]))

        result_all_dict = {}

        for model_name, model_tupple in self.model_dict.items():
            if verbose:
                print('Training model {}'.format(model_name), flush=True)
            result_dict, cv_score_dict = self.run_single_model(model_name, n_iters, n_jobs)
            result_all_dict[model_name] = result_dict
            
            if verbose:
                print('{}: \nmean val rmse: {} \nmean test rmse: {}' \
                      '\nmean val rsquare: {} \nmean test rsquare: {}'.format(model_name,
                                                                              result_dict['mean_val_rmse'],
                                                                              result_dict['mean_test_rmse'],
                                                                              result_dict['mean_val_rsquare'],
                                                                              result_dict['mean_test_rsquare']),
                      flush=True)

        df_summary = pd.DataFrame(result_all_dict).T

        return df_summary