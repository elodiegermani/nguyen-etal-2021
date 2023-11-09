import pandas as pd
import os

def save_features(
    pipeline:str, 
    features_dict:dict, 
    outcome_dict:dict, 
    specific:str='', 
    features_dir:str='features'):
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
            for feature in ['alff', 'falff', 'ReHo', 'zalff', 'zfalff', 'zReHo']:
                
                df_features = features_dict[timepoint]
                df_outcome = outcome_dict[timepoint]
    
                if pipeline != 'no_imaging_features':
                    subjects_cohort = [int(i) for i in df_features.index.tolist()]
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