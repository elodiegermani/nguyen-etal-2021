import pandas as pd
import numpy as np
import os
from ..utils.cohort_utils import load_ppmi_csv
import livingpark_utils
import datetime as dt


from ..constants import (
    FILENAME_PARTICIPANT_STATUS,
    FILENAME_DEMOGRAPHICS,
    FILENAME_PD_HISTORY,
    FILENAME_AGE,
    FILENAME_SOCIO,
    FILENAME_MOCA,
    FILENAME_UPDRS1A,
    FILENAME_UPDRS1B,
    FILENAME_UPDRS2,
    FILENAME_UPDRS3,
    FILENAME_UPDRS3_CLEAN,
    FILENAME_UPDRS4,
    FILENAME_GDS,
    FILENAME_DOMSIDE,
    FILENAME_FMRI_INFO,
    FILENAME_FMRI_INFO_ZIP,
    FILENAME_FMRI_METADATA
)

from ..constants import (
    COL_PAT_ID,
    COL_VISIT_TYPE,
    COL_STATUS,
    COL_PD_STATE,
    COL_AGE,
    COL_SEX,
    COL_EDUCATION,
    COL_UPDRS3,
    COL_UPDRS1A,
    COL_UPDRS1B,
    COL_UPDRS1,
    COL_UPDRS2,
    COL_UPDRS4,
    COL_MOCA,
    COL_IMAGING_PROTOCOL,
    COL_IMAGE_DESC
)

from ..constants import (
    COL_DATE_INFO,
    COL_DATE_BIRTH,
    COL_DATE_PD,
    FIELD_STRENGTH,
    MANUFACTURER,
    IMAGE_DESC,
    STATUS_PD,
    TR_VAL,
)

from ..constants import (
    COL_IMAGING_PROTOCOL,
    COLS_DATE, 
    IDA_STATUS_MAP,
    IDA_COLNAME_MAP,
    IDA_VISIT_MAP,
    STATUS_MED,
)

# How to deal with NaNs ? 
def impute_mean(
    df:pd.DataFrame, 
    col_name:str, 
    is_int:bool=False
) -> pd.DataFrame:
    """Impute missing values with mean values.

    Parameters
    ----------
    df : pd.DataFrame
        input dataframe
    col_name : str | list
        columns to impute
    is_int : bool
        return the int mean value if column contains only int values

    Returns
    -------
    pd.DataFrame
        dataframe with imputed missing values
    """
    mean_value = df[col_name].mean()
    if is_int:
        mean_value = int(mean_value)
    df[col_name].fillna(value=mean_value, inplace=True)
    
    return df

def impute_zeros(
    df:pd.DataFrame, 
    col_name:str
) -> pd.DataFrame:
    """Impute missing values with 0 values.

    Parameters
    ----------
    df : pd.DataFrame
        input dataframe
    col_name : str | list
        columns to impute

    Returns
    -------
    pd.DataFrame
        dataframe with imputed missing values
    """

    df[col_name].fillna(value=0, inplace=True)
    
    return df

def get_features(
    df_baseline:pd.DataFrame, 
    utils:livingpark_utils.LivingParkUtils, 
    participants_list:list, 
    timepoint:str='baseline', 
    add_DOMSIDE:bool=False
) -> pd.DataFrame:
    """Gather features to train machine learning models. 

    Parameters
    ----------
    df_baseline : pd.DataFrame
        input dataframe with baseline values
    utils : livingpark_utils.LivingParkUtils
        the notebook's LivingParkUtils instance
    participants_list: list
        list of participants to get features
    timepoint : str
        timepoint of the cohort to compute summary measures (one of 'baseline', '1y', '2y' and '4y')
    add_DOMSIDE : bool
        whether to include DOMSIDE column in the features

    Returns
    -------
    pd.DataFrame
        dataframe with all features
    """
    # Load necessary study files
    df_pd_history = load_ppmi_csv(utils, FILENAME_PD_HISTORY)
    df_demographics = load_ppmi_csv(utils, FILENAME_DEMOGRAPHICS)
    df_socio = load_ppmi_csv(utils, FILENAME_SOCIO)
    df_age = load_ppmi_csv(utils, FILENAME_AGE)

    df_baseline = df_baseline[df_baseline[COL_PAT_ID].isin(participants_list)]
    df_features = df_baseline[['PATNO', 'EVENT_ID','MCATOT', 'GDS_TOTAL', 'UPDRS_TOT']]
    
    # Necessary columns for demographic file
    cols_demo = ['PATNO', 'SEX', 'RAWHITE', 'HISPLAT', 'RAINDALS','RABLACK', 'RAASIAN', 
                 'RAHAWOPI', 'RANOS','HANDED']
    
    df_features = df_features.merge(
        df_demographics[cols_demo],
        on=[COL_PAT_ID],
    )

    # Encode HANDED column
    df_features['HANDED_RIGHT'] = df_features['HANDED'].apply(lambda x: x == 1)
    df_features['HANDED_LEFT'] = df_features['HANDED'].apply(lambda x: x == 2)
    df_features['HANDED_BOTH'] = df_features['HANDED'].apply(lambda x: x == 3)
    df_features = df_features.drop(['HANDED'], axis=1)

    # Necessary columns for PD file
    cols_PD = ['PATNO','INFODT','SXDT', 'PDDXDT', 'DXTREMOR', 'DXRIGID', 
               'DXBRADY', 'DXPOSINS']
    
    df_features = df_features.merge(
        df_pd_history[cols_PD],
        on=[COL_PAT_ID],
    )

    if add_DOMSIDE: # Archived feature, only use if mentioned 
        df_domside = load_ppmi_csv(utils, FILENAME_DOMSIDE)
        cols_DOMSIDE = ['PATNO', 'DOMSIDE']
        df_features = df_features.merge(
            df_domside[cols_DOMSIDE],
            on=[COL_PAT_ID],
        )
        # Encode DOMSIDE column
        df_features['DOMSIDE_LEFT'] = df_features['DOMSIDE'].apply(lambda x: (x==1) if not pd.isnull(x) else np.nan)
        df_features['DOMSIDE_RIGHT'] = df_features['DOMSIDE'].apply(lambda x: (x == 2) if not pd.isnull(x) else np.nan)
        df_features['DOMSIDE_BOTH'] = df_features['DOMSIDE'].apply(lambda x: (x == 3) if not pd.isnull(x) else np.nan)
        df_features = df_features.drop(['DOMSIDE'], axis=1)
    
    # Necessary columns for social file 
    cols_socio = ['PATNO','EDUCYRS']
    
    df_features = df_features.merge(
        df_socio[cols_socio].drop_duplicates(subset=[COL_PAT_ID]),
        on=[COL_PAT_ID],
    )

    # Necessary columns for age file
    df_features = df_features.merge(df_age, on=[COL_PAT_ID, COL_VISIT_TYPE])
    df_features = df_features.drop_duplicates(subset=[COL_PAT_ID])
    
    df_features['INFODT'] = pd.to_datetime(df_features['INFODT'], format='%m-%y')
    df_features['PDDXDT'] = pd.to_datetime(df_features['PDDXDT'], format='%m-%y')
    
    # Manage dates with SXDT columns (M/Y to M-Y)
    list_symptom_date = []
    for i, row in df_features.iterrows():
        date = row['SXDT']
        m = int(date.split('/')[0])
        y = int(date.split('/')[1])
        list_symptom_date += [dt.datetime.strptime('{:02d}-{}'.format(int(m), int(y)), '%m-%Y')]
    
    df_features['SXDT'] = list_symptom_date
    
    # Days bw diagnosis and visit
    df_features['V-DXDT'] = (
            df_features[COL_DATE_INFO] - df_features[COL_DATE_PD]) / np.timedelta64(1, 'D')
    
    # Days bw symptoms and visit
    df_features['V-SXDT'] = (
            df_features[COL_DATE_INFO] - df_features['SXDT']) / np.timedelta64(1, 'D')
    
    df_features = df_features.drop(['INFODT', 'SXDT', 'PDDXDT'],axis=1)
    
    if timepoint == 'baseline':
        df_features = df_features.drop(['UPDRS_TOT'], axis=1) # only included for prediction
        
    df_features = df_features.set_index('PATNO')
    
    df_features = impute_mean(df_features, 'MCATOT', True)
    df_features = impute_mean(df_features, 'GDS_TOTAL', True)
    
    return df_features

def get_threshold(
    df_cohort_baseline:pd.DataFrame, 
    df_cohort_1y:pd.DataFrame,
    df_cohort_2y:pd.DataFrame, 
    df_cohort_4y:pd.DataFrame
) -> int:
    """Compute threshold to define high and low-severity groups.

    Parameters
    ----------
    df_cohort_baseline : pd.DataFrame
        input dataframe with baseline values
    df_cohort_1y : pd.DataFrame
        input dataframe with 1y values
    df_cohort_2y : pd.DataFrame
        input dataframe with 2y values
    df_cohort_4y : pd.DataFrame
        input dataframe with 4y values

    Returns
    -------
    int
        threshold representing the mean value of all UPDRS score across years and participants.
    """
    df_longitudinal_scores = df_cohort_baseline[['PATNO', 'EVENT_ID', 'NP3TOT' ,'UPDRS_TOT']].copy()
    for i, df_pred in enumerate([df_cohort_1y, df_cohort_2y, df_cohort_4y]):
        if i < 2: 
            y = i+1
        else:
            y = 4
        df_longitudinal_scores[f'UPDRS_TOT_{y}Y'] = [df_pred['UPDRS_TOT'][df_pred['PATNO']==sub_id].tolist()[0] \
                                             if sub_id in df_pred['PATNO'].tolist() else np.nan \
                                              for sub_id in df_cohort_baseline['PATNO'].tolist()]
    
        df_longitudinal_scores[f'NP3TOT_{y}Y'] = [df_pred['NP3TOT'][df_pred['PATNO']==sub_id].tolist()[0] \
                                              if sub_id in df_pred['PATNO'].tolist() else np.nan \
                                              for sub_id in df_cohort_baseline['PATNO'].tolist()]

    threshold = int(df_longitudinal_scores[['UPDRS_TOT', 'UPDRS_TOT_1Y', 'UPDRS_TOT_2Y', 'UPDRS_TOT_4Y']].median().mean())
    
    return threshold

def get_outcome_measures(
    df:pd.DataFrame, 
    threshold:int
) -> pd.DataFrame:

    """Gather target score for the machine learning models

    Parameters
    ----------
    df : pd.DataFrame
        input dataframe with features
    threshold : int
        threshold to define high and low-severity groups.

    Returns
    -------
    pd.DataFrame
        dataframe with outcome measures.
    """
    df_outcome = df[['PATNO', 'EVENT_ID', 'NP3TOT' ,'UPDRS_TOT']].copy()
    df_outcome['SEVERITY'] = [1 if df_outcome['UPDRS_TOT'].iloc[i] > threshold else 0 for i in range(len(df_outcome))]
    
    return df_outcome

def save_features(
    pipeline:str, 
    features_dict:dict, 
    outcome_dict:dict, 
    specific:str='', 
    features_dir:str='features',
    features_list:list=['zfalff', 'zReHo']):
    '''
    Save features for training. 

    Parameters
    ----------

    pipeline: str
        name of the pipeline used 

    features_dict: dict of pd.DataFrame
        dict. of features to save with keys=timepoint, value=dataframe.

    outcome_dict: dict of pd.DataFrame
        outcome target

    specific: str
        if variation was applied

    features_dir: str
        where imaging features dataframe are stored.
    '''

    for timepoint in list(features_dict.keys()):
        for atlas in ['schaefer', 'basc197', 'basc444']:
            for feature in features_list:
                
                df_features = features_dict[timepoint]
                df_outcome = outcome_dict[timepoint]
    
                if pipeline != 'no_imaging_features':
                    subjects_cohort = [int(i) for i in df_features.index.tolist()]

                    if specific =='unclean':
                        for i in subjects_cohort:
                            if subjects_cohort[i]==3125:
                                subjects_cohort[i]='3125b'

                    file_list = [f'./outputs/{pipeline}/{features_dir}/sub-{sub}/sub-{sub}_{feature}_atlas-{atlas}.csv'\
                                 for sub in subjects_cohort]
                    
                    df_img_features = pd.DataFrame()
                    for f in file_list: 
                        df = pd.read_csv(f, header=0, index_col=0)
                        df_img_features = pd.concat([df_img_features, df])
                    
                    df_img_features.index = df_features.index
        
                    df_all_features = pd.merge(df_img_features, df_features, 
                                        left_index=True, right_index=True)
                    
                    output_dir=f'./outputs/{pipeline}/prediction_scores/prediction-{timepoint}'+\
                    f'_atlas-{atlas}_feature-{feature}'

                    if 'only-imaging' in specific: 
                        df_all_features = df_img_features

                else: 
                    df_all_features = df_features
                    output_dir=f'./outputs/{pipeline}/prediction_scores/prediction-{timepoint}'
                
                if specific:
                    output_dir += specific

                # Save features
                if not os.path.isdir(output_dir):
                    os.mkdir(output_dir)

                if not os.path.exists(f'{output_dir}/data.csv') or not os.path.exists(f'{output_dir}/target.csv'):
                    df_all_features.to_csv(f'{output_dir}/data.csv',
                                       header=True, index=False)
                    df_outcome.to_csv(f'{output_dir}/target.csv',
                                       header=True, index=False)