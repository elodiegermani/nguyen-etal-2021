# Train machine learning models 
from sklearn import model_selection, pipeline, preprocessing, impute, feature_selection, metrics, svm, linear_model, \
	ensemble
import copy
from nguyenetal.prediction import cross_validation, regression_model
import pandas as pd
import numpy as np

def get_permutation_score(pipeline, specific, timepoint, atlas, feature, model, n_iters=100, n_jobs=32):

    outer = model_selection.LeaveOneOut()
    inner = cross_validation.StratifiedKFoldContinuous(n_splits=10, n_bins=3, 
                                                       shuffle=True, random_state=989)   

    output_dir=f'./outputs/{pipeline}/prediction_scores/'+\
    f'prediction-{timepoint}_atlas-{atlas}_feature-{feature}'

    if pipeline == 'no_imaging_features':
        output_dir=f'./outputs/{pipeline}/prediction_scores/'+\
    f'prediction-{timepoint}'

    if specific:
        output_dir += specific

    df_all_features = pd.read_csv(f'{output_dir}/data.csv',
                       header=0, index_col=None)
    df_outcome = pd.read_csv(f'{output_dir}/target.csv',
                       header=0, index_col=None)

    target = df_outcome['UPDRS_TOT'].to_numpy(copy=True)

    if 'EVENT_ID' in df_all_features.columns:
        data = df_all_features.drop(['EVENT_ID'], axis=1).astype(np.float64)

    regressors = regression_model.RegressorPanel(data,target,
            outer=outer,
            inner=inner,
            output_dir=output_dir)

    regressors.set_feature_selection(None)
    
    score, perm_scores, pvalue = regressors.run_single_model_permutation(model, n_iters=n_iters, n_jobs=n_jobs)

    scores_dict = {'score': [score], 'pvalue': [pvalue]}

    df_score = pd.DataFrame(scores_dict)
    df_score.to_csv(f'{output_dir}/permutation_scores_{model}.csv')

