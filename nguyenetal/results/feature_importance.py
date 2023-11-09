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
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import image, plotting, datasets
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
'HANDED_RIGHT': 'Right handed',
'HANDED_LEFT': 'Left handed',
'HANDED_BOTH': 'Ambidextrous',
'V-DXDT': 'Time since diag.',
'V-SXDT': 'Symptom duration',
'DXTREMOR': 'Tremor',
'DXRIGID': 'Rigidity',
'DXBRADY': 'Bradykinesia',
'DXPOSINS': 'Postural Instability',
'MCATOT': 'Baseline MoCA',
'GDS_TOTAL': 'Baseline GDS',
'UPDRS_TOT': 'Baseline MDS-UPDRS',
'DOMSIDE_RIGHT': 'Dominant right',
'DOMSIDE_LEFT': 'Dominant left',
'DOMSIDE_BOTH': 'Dominant both'}

def get_features_importance(
    model_file:str, 
    roi_labels:list,
    target:pd.DataFrame, 
    data:pd.DataFrame, 
    importance_attr:str='coef_'):

    '''
    Computes the dataframe with feature importance to be plotted after. 

    Parameters
    ----------

    model_file:str
        path to model saved after training

    roi_labels: list
        list of labels for brain regions (computed using utils/atlas_labels.py)

    target: pd.DataFrame
        outcome measurements

    data: pd.DataFrame
        features

    importance_attr:str
        whether to use coef or gini.

    Returns
    -------

    df_importance: pd.DataFrame
        dataframe with values of feature importance

    '''
    
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
    # Compute median feature importance across all splits for each feature ?
    df_importance['Feature Importance'] = np.abs(np.median(features_array, 0))
    # Divide by the max value of the feature importance ???
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

def plot_features_importance(
    df_importance:pd.DataFrame, 
    n_features_plot:int
    ):

    '''
    Plot the features

    Parameters
    ----------
    df_importance:pd.DataFrame
        dataframe of features and importance value

    n_features_plot:int
        n. of features to plot.
    '''

    fig, axis = plt.subplots(1, 1, figsize=(8, int(0.5*n_features_plot)))
    bars = sns.barplot(y='Feature', x='Feature Importance', hue='Correlation',
                       data=df_importance.sort_values('Feature Importance', 
                                                      ascending=False).iloc[:n_features_plot],
                       ax=axis, dodge=False,
                       palette=sns.xkcd_palette(['pale red', 'windows blue']), saturation=0.75,
                       hue_order=['positive', 'negative'],
                       orient='h')
    bars.tick_params(labelsize=16)
    plt.yticks(va='top') 
    axis.set_ylabel('Feature',fontsize=16, fontweight='bold')
    axis.set_xlabel('Importance',fontsize=16, fontweight='bold')

    plt.legend(loc='lower right')
    plt.tight_layout()

    return fig

def _values_to_img(
    val_array:np.array, 
    atlas_img:nib.Nifti1Image):
    '''
    Build an image of the atlas with values associated with the feature importance of this region. 

    Parameters
    ----------
        val_array:np.array
            Array of feature value. 

        atlas_img:nib.Nifti1Image
            Image of the atlas.

    Returns
    -------
        nib.Nifti1Image
            image with values for brain regions replaced with feature importance. 

    '''
    atlas_array = atlas_img.get_fdata()
    val_img_array = np.zeros_like(atlas_array, dtype=np.float64)

    if val_array.size != (atlas_array.max()):
        raise ValueError('There are {} values and {} ROIs'.format(val_array.size, atlas_array.max()))

    for i in range(val_array.size):
        val_img_array[atlas_array == (i + 1)] = val_array[i]

    return image.new_img_like(data=val_img_array, ref_niimg=atlas_img)

def plot_brain_features_importance(
    df_importance, 
    roi_labels, 
    atlas, 
    n_features):
    '''
    Parameters
    ----------
    df_importance:pd.DataFrame
        dataframe of features and importance value

    n_features:int
        n. of features to plot.

    roi_labels: list
        list of labels for brain regions (computed using utils/atlas_labels.py)
    
    atlas: str
        which atlas to use. one of ['basc197','basc444', 'schaefer']

    Returns
    -------

    fig: plt.Figure
        figure with brain feature importance

    '''
    basc197_atlas = datasets.fetch_atlas_basc_multiscale_2015(version="sym", resolution=197)['map']
    basc444_atlas = datasets.fetch_atlas_basc_multiscale_2015(version="sym", resolution=444)['map']
    schaefer_atlas = datasets.fetch_atlas_schaefer_2018(100, resolution_mm=2)['maps']
    cerebellar_atlas = f'./inputs/atlases/Cerebellum-MNIfnirt-maxprob-thr25-2mm.nii.gz'
    striatum_atlas = './inputs/atlases/striatum-con-label-thr25-7sub-2mm.nii.gz'
    
    atlas_dict = {
        'basc197':basc197_atlas,
        'basc444': basc444_atlas,
        'schaefer': [schaefer_atlas, cerebellar_atlas, striatum_atlas]
    }

    if atlas == 'schaefer':
        cerebellar_data = nib.load(cerebellar_atlas).get_fdata()
        cerebellar_data[cerebellar_data != 0] += 100
        cerebellar_data[nib.load(schaefer_atlas).get_fdata() != 0] = 0
        striatum_data = nib.load(striatum_atlas).get_fdata()
        striatum_data[striatum_data != 0] += 128
        striatum_data[nib.load(schaefer_atlas).get_fdata() != 0] = 0
        atlas_data = nib.load(schaefer_atlas).get_fdata() + cerebellar_data + striatum_data
        atlas_img = nib.Nifti1Image(atlas_data, nib.load(schaefer_atlas).affine)
    else:
        atlas_img = nib.load(atlas_dict[atlas])

    val_atlas_array = df_importance['Feature Importance'].loc[(df_importance['Feature'].isin(roi_labels))].values
    val_atlas_array[np.argsort(val_atlas_array)[:-n_features]] = 0

    neg_array = np.array(df_importance['Correlation'].loc[(df_importance['Feature'].isin(roi_labels))].values.flatten() == 'negative')
    val_atlas_array[neg_array] *= -1

    img_importance = _values_to_img(val_atlas_array, atlas_img)
    fig, ax = plt.subplots(3, 1, figsize=(8, 5))
    brainX = plotting.plot_stat_map(img_importance, cmap='coolwarm', display_mode='x', cut_coords=4, alpha=0.95, axes=ax[0], vmax=1)
    brainY = plotting.plot_stat_map(img_importance, cmap='coolwarm', display_mode='y', cut_coords=4, alpha=0.95, axes=ax[1], vmax=1)
    brainZ = plotting.plot_stat_map(img_importance, cmap='coolwarm', display_mode='z', cut_coords=4, alpha=0.95, axes=ax[2], vmax=1)
    plt.tight_layout()

    return fig

timepoint_dict = {'baseline':'Baseline',
                 '1y': 'Year 1',
                 '2y': 'Year 2', 
                 '4y': 'Year 4'}

feature_dict = {'zfalff':'fALFF',
               'zReHo':'ReHo'}

atlas_dict = {'schaefer':'Schaefer', 
             'basc197':'BASC197',
             'basc444': 'BASC444'}

model_dict = {'ElasticNet':'ElasticNet',
             'LinearSVR': 'SVM',
             'GradientBoostingRegressor': 'Gradient Boosting',
             'RandomForestRegressor': 'Random Forest'}

def plot_all_feature_importance(
    global_df:pd.DataFrame, 
    pipeline:str, 
    specific:str):

    '''
    Parameters
    ----------
    global_df:pd.DataFrame
        dataframe of prediction scores to find best model and best atlas

    pipeline: str

    specific: str
    '''
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
            f'prediction-{timepoint}_atlas-{best_atlas}_feature-{feature}'

            if specific:
                output_dir += specific

            if pipeline != 'no_imaging_features':
                with open(f'./inputs/atlases/{best_atlas}_labels.txt', 'r') as f:
                    roi_labels = f.readlines()

            else:
                roi_labels = []
                output_dir=f'./outputs/{pipeline}/prediction_scores/'+\
            f'prediction-{timepoint}'
                best_atlas = 'none'

            model_file = f'{output_dir}/{best_model}_results.pkl'

            df_all_features = pd.read_csv(f'{output_dir}/data.csv',
                               header=0, index_col=None)
            df_outcome = pd.read_csv(f'{output_dir}/target.csv',
                               header=0, index_col=None)

            target = df_outcome['UPDRS_TOT']#.to_numpy(copy=True)

            if 'EVENT_ID' in df_all_features.columns:
                df_all_features = df_all_features.drop(['EVENT_ID'], axis=1)
            data = df_all_features.astype(np.float64)

            if best_model == 'ElasticNet' or best_model=='LinearSVR':
                importance_attr = 'coef_'
            else: 
                importance_attr = 'feature_importances_'

            df_importance = get_features_importance(model_file, roi_labels,
                              target, data, importance_attr=importance_attr,
                              )

            fig = plot_features_importance(df_importance,n_features_plot=10)
            if specific != None:
                plt.savefig(f'./outputs/{pipeline}/figures/feature_importance_prediction-{timepoint}_feature-{feature}_model-{best_model}_atlas-{best_atlas}{specific}.png')
            else: 
                plt.savefig(f'./outputs/{pipeline}/figures/feature_importance_prediction-{timepoint}_feature-{feature}_model-{best_model}_atlas-{best_atlas}.png')
            
            if pipeline != 'no_imaging_features':
                fig_2 = plot_brain_features_importance(df_importance, roi_labels, best_atlas, n_features=10)
                if specific != None:
                    plt.savefig(f'./outputs/{pipeline}/figures/feature_importance_maps_prediction-{timepoint}_feature-{feature}_model-{best_model}_atlas-{best_atlas}{specific}.png')
                else:
                    plt.savefig(f'./outputs/{pipeline}/figures/feature_importance_maps_prediction-{timepoint}_feature-{feature}_model-{best_model}_atlas-{best_atlas}.png')