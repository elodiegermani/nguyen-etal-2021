# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=RuntimeWarning)

import pickle, sys, os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from nilearn import image, plotting
pd.options.mode.chained_assignment = None

dict_clin_labels = {'AGE_AT_VISIT': 'Age',
'SEX': 'Male',
'HISPLAT': 'Hispanic',
'RAINDALS': 'Am. Indian/Alask. Nat.',
'RAASIAN': 'Asian',
'RABLACK': 'Afr-Amer.',
'RAHAWOPI': 'Pac. Isl.',
'RAWHITE': 'Caucasian',
'RANOS': 'Other race',
'EDUCYRS': 'Years of edu.',
'HANDED': 'Handed',
'V-DXDT': 'Time since diag.',
'V-SXDT': 'Symptom duration',
'DXTREMOR': 'Tremor',
'DXRIGID': 'Rigidity',
'DXBRADY': 'Bradykinesia',
'DXPOSINS': 'Postural Instability',
'MCATOT': 'Baseline MoCA',
'GDS_TOTAL': 'Baseline GDS',
'UPDRS_TOT': 'Baseline MDS-UPDRS'}

def get_features_importance(model_file, roi_labels,
                  target, data, b_delta=False, importance_attr='coef_'):
    
    n_rois = len(roi_labels) 
    clin_labels = [dict_clin_labels[s] for s in data.columns[n_rois:]] # Only clinical labels
    data.columns = np.concatenate((roi_labels, clin_labels)) # Change column names with real ROIs labels and clinical cleaned 

    with open(model_file, 'rb') as f:
        model_dict = pickle.load(f) # Load model

    n_splits = len(model_dict['estimator']) 
    n_features = getattr(model_dict['estimator'][0].best_estimator_.steps[-1][1], 
                         importance_attr).shape[-1]
    features_array = np.zeros((n_splits, n_features)) # Create features array 
    
    for i in range(len(model_dict['estimator'])):
        model = model_dict['estimator'][i].best_estimator_.steps[-1][1]
        features_array[i,:] = getattr(model, importance_attr)

    df_importance = pd.DataFrame(index=data.columns, columns=['Median Feature Importance', 'Correlation'])
    df_importance['Feature Importance'] = np.abs(np.median(features_array, 0))
    df_importance['Feature Importance'] /= df_importance['Feature Importance'].abs().max()
    
    if importance_attr == 'feature_importances_':
        for n in range(df_importance.shape[0]):
            feat_array = data.iloc[:, n].values
            arr_nan = np.isnan(feat_array) | np.isnan(target.astype(np.float64).values.flatten())
            r, p = stats.pearsonr(target.values.flatten()[~arr_nan], feat_array.astype(np.float64)[~arr_nan])

            df_importance['Correlation'].iloc[n] = 'negative' if r < 0 else 'positive'
    else:
        df_importance['Correlation'] = ['positive' if a > 0 else 'negative' for a in np.median(features_array, 0)]
    
    df_importance['Feature'] = df_importance.index 

    return df_importance 

def plot_features_importance(df_importance, n_features_plot):
    fig, axis = plt.subplots(1, 1, figsize=(8, int(0.5*n_features_plot)))
    bars = sns.barplot(y='Feature', x='Feature Importance', hue='Correlation',
                       data=df_importance.sort_values('Feature Importance', 
                                                      ascending=False).iloc[:n_features_plot],
                       ax=axis, dodge=False,
                       palette=sns.xkcd_palette(['pale red', 'windows blue']), saturation=0.75,
                       hue_order=['positive', 'negative'],
                       orient='h')

    plt.legend(loc='lower right')

    return fig

timepoint_dict = {'baseline':'Baseline',
                 '1y': 'Year 1',
                 '2y': 'Year 2', 
                 '4y': 'Year 4'}

feature_dict = {'falff':'fALFF',
               'ReHo':'ReHo'}

atlas_dict = {'schaefer':'Schaefer', 
             'basc197':'BASC197',
             'basc444': 'BASC444'}

model_dict = {'ElasticNet':'ElasticNet',
             'LinearSVR': 'SVM',
             'GradientBoostingRegressor': 'Gradient Boosting',
             'RandomForestRegressor': 'Random Forest'}

def plot_all_feature_importance(global_df, pipeline):
    for timepoint, t in timepoint_dict.items():
        for feature, f in feature_dict.items():

            best_model = global_df['Best performing model'].loc[(global_df['Type']=='Replication') &\
                                    (global_df['Feature']==feature_dict[feature]) &\
                                    (global_df['MDS-UPDRS Prediction target']==timepoint_dict[timepoint])].iloc[0]
    
            best_model = list(model_dict.keys())[list(model_dict.values()).index(best_model)]
    
            best_atlas = global_df['Best performing parcellation'].loc[(global_df['Type']=='Replication') &\
                                    (global_df['Feature']==feature_dict[feature]) &\
                                    (global_df['MDS-UPDRS Prediction target']==timepoint_dict[timepoint])].iloc[0].lower()
            output_dir=f'./outputs/{pipeline}/prediction_scores/'+\
            f'predition-{timepoint}_atlas-{best_atlas}_feature-{feature}'

            model_file = f'{output_dir}/{best_model}_results.pkl'

            with open(f'./inputs/atlases/{best_atlas}_labels.txt', 'r') as f:
                roi_labels = f.readlines()

            df_all_features = pd.read_csv(f'{output_dir}/data.csv',
                               header=0, index_col=None)
            df_outcome = pd.read_csv(f'{output_dir}/target.csv',
                               header=0, index_col=None)

            target = df_outcome['UPDRS_TOT']#.to_numpy(copy=True)
            data = df_all_features.drop(['EVENT_ID'], axis=1).astype(np.float64)

            if best_model == 'ElasticNet' or best_model=='LinearSVR':
                importance_attr = 'coef_'
            else: 
                importance_attr = 'feature_importances_'

            df_importance = get_features_importance(model_file, roi_labels,
                              target, data, b_delta=False, importance_attr=importance_attr,
                              )

            fig = plot_features_importance(df_importance,n_features_plot=10)

            plt.savefig(f'./outputs/{pipeline}/figures/feature_importance_prediction-{timepoint}_feature-{feature}_model-{best_model}_atlas-{best_atlas}.png')