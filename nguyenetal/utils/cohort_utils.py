"""Helper functions for Nguyen et al. notebooks."""
from pathlib import Path

import pandas as pd
import numpy as np
import datetime as dt

import livingpark_utils
from livingpark_utils.scripts import pd_status
from livingpark_utils.dataset.ppmi import disease_duration

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

def convert_date_cols(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    """Convert date columns from str to pandas datetime type.

    Parameters
    ----------
    df : pd.DataFrame
        input dataframe
    cols : list, optional
        list of date columns to convert

    Returns
    -------
    pd.DataFrame
        dataframe with converted columns
    """
    if cols is None:
        cols = COLS_DATE
    for col in cols:
        try:
            df[col] = pd.to_datetime(df[col])
        except KeyError:
            continue
    return df

def mean_impute(df: pd.DataFrame, cols: str | list) -> pd.DataFrame:
    """Impute missing values with the mean.

    Parameters
    ----------
    df : pd.DataFrame
        input dataframe
    col : str | list
        columns to impute

    Returns
    -------
    pd.DataFrame
        dataframe with imputed missing values
    """
    if type(cols) == str:
        cols = [cols]

    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col].isna(), col] = df[col].mean()

    return df

def zeros_impute(df: pd.DataFrame, cols: str | list) -> pd.DataFrame:
    """Impute missing values with 0 values.

    Parameters
    ----------
    df : pd.DataFrame
        input dataframe
    col : str | list
        columns to impute

    Returns
    -------
    pd.DataFrame
        dataframe with imputed missing values
    """
    if type(cols) == str:
        cols = [cols]

    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col].isna(), col] = 0

    return df

def load_ppmi_csv(
    utils: livingpark_utils.LivingParkUtils,
    filename: str,
    from_ida_search: bool = False,
    convert_dates: bool = True,
    alternative_dir: str = ".",
    cols_to_impute: str | list | None = None,
    convert_int: str | list | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Load PPMI csv file as a pandas dataframe.

    Parameters
    ----------
    utils : livingpark_utils.LivingParkUtils
        the notebook's LivingParkUtils instance
    filename : str
        name of file to be loaded
    from_ida_search : bool, optional
        if True, column names and values will be converted from IDA format
        to match the other PPMI study files, by default False
    convert_dates : bool, optional
        if True, date columns will be converted to pd.datetime format, by default True
    alternative_dir : str, optional
        fallback directory if file is not found in utils.study_files_dir, by default "."
    cols_to_impute : str | list, optional
        column(s) where missing values should be imputed with the mean, by default None
    **kwargs : optional
        additional keyword arguments to be passed to convert_date_cols()

    Returns
    -------
    pd.DataFrame
        loaded/preprocessed dataframe

    Raises
    ------
    FileNotFoundError
        file not found in either utils.study_files_dir or alternative_dir
    RuntimeError
        IDA format conversion issue
    """
    filepath = Path(utils.study_files_dir, filename)

    if not filepath.exists():
        filepath = Path(alternative_dir, filename)
    if not filepath.exists():
        raise FileNotFoundError(
            f"File {filename} is not in either "
            f"{utils.study_files_dir} or {alternative_dir}"
        )
    df_ppmi = pd.read_csv(filepath)

    # convert IDA search results to the same format as other PPMI study files
    if from_ida_search:
        df_ppmi = df_ppmi.rename(columns=IDA_COLNAME_MAP)

        # convert visit code
        #missing_keys = set(df_ppmi[COL_VISIT_TYPE]) - set(IDA_VISIT_MAP.keys())
        #if len(missing_keys) != 0:
        #    raise RuntimeError(f"Missing keys in conversion map: {missing_keys}")
        df_ppmi[COL_VISIT_TYPE] = df_ppmi[COL_VISIT_TYPE].map(IDA_VISIT_MAP)
        #df_ppmi[COL_STATUS] = df_ppmi[COL_STATUS].map(IDA_STATUS_MAP)

    # convert subject IDs to integers
    # IDs should be all integers to be consistent
    # if they are strings the cohort_ID hash can change even if the strings are the same
    df_ppmi[COL_PAT_ID] = pd.to_numeric(df_ppmi[COL_PAT_ID], errors="coerce")
    
    if convert_int:
        for cols in convert_int:
            df_ppmi[cols] = pd.to_numeric(df_ppmi[cols], errors="coerce")
    
    invalid_subjects = df_ppmi.loc[df_ppmi[COL_PAT_ID].isna(), COL_PAT_ID].to_list()
    if len(invalid_subjects) > 0:
        print(
            f"Dropping {len(invalid_subjects)} subjects with non-integer IDs"
            # f": {invalid_subjects}"
        )
        df_ppmi = df_ppmi.loc[~df_ppmi[COL_PAT_ID].isin(invalid_subjects)]

    if convert_dates:
        df_ppmi = convert_date_cols(df_ppmi, **kwargs)

    if cols_to_impute is not None:
        df_ppmi = zeros_impute(df_ppmi, cols_to_impute)

    return df_ppmi

def get_fMRI_cohort(
    utils: livingpark_utils.LivingParkUtils,
    filename: str = FILENAME_FMRI_METADATA,
) -> pd.DataFrame:
    """Extract base main cohort for Nguyen et al. papers.

    Parameters
    ----------
    utils : livingpark_utils.LivingParkUtils
        the notebook's LivingParkUtils instance
    filename : str
        name of FMRI search result file, by default FILENAME_FMRI_INFO_XLSX.

    Returns
    -------
    pd.DataFrame
        dataframe with FMRI information for selected cohort

    Raises
    ------
    ValueError
        invalid value for cohort_name parameter
    RuntimeError
        duplicate subjects in output dataframe
    """
    
    field_strength = FIELD_STRENGTH
    status_groups = STATUS_PD
    status_med = STATUS_MED

    dirname = utils.study_files_dir
    filepath = Path(dirname, filename)

    # error if file doesn't exist yet
    if not filepath.exists():
        raise FileNotFoundError(
            f"{filepath} doesn't exist. "
            "You need to run livingpark_utils.ppmi.Downloader.get_study_file first"
        )
    else:
        print(f"Using fMRI info file: {filepath}")

    # load csv files
    df_fMRI = load_ppmi_csv(utils, FILENAME_FMRI_METADATA, from_ida_search=True)
    df_fMRI = df_fMRI.drop_duplicates()  # some rows are identical
    df_status = load_ppmi_csv(utils, FILENAME_PARTICIPANT_STATUS)

    # drop subjects with NA for ID (conversion to numerical value failed)
    df_fMRI_subset = df_fMRI
    df_fMRI_subset = df_fMRI_subset.dropna(axis="index", subset=[COL_PAT_ID])
    # only keep PD patients
    df_fMRI_subset = df_fMRI_subset.merge(df_status, on=[COL_PAT_ID])

    df_fMRI_subset = df_fMRI_subset.loc[
        (df_fMRI_subset[COL_STATUS].isin([status_groups])) & \
        (df_fMRI_subset[COL_IMAGING_PROTOCOL].str.contains(MANUFACTURER)) & \
        (df_fMRI_subset[COL_IMAGING_PROTOCOL].str.contains(FIELD_STRENGTH)) & \
        #(df_fMRI_subset[COL_IMAGING_PROTOCOL].str.contains(TR_VAL)) & \
        (df_fMRI_subset[COL_IMAGE_DESC].isin(IMAGE_DESC))]

    #df_fMRI_subset = df_fMRI_subset.loc[
    #    (df_fMRI_subset[COL_FMRI]==1)]

    # warn if there are duplicates
    if df_fMRI_subset[COL_PAT_ID].nunique() != len(df_fMRI_subset[COL_PAT_ID]):
        print(f"WARNING: Duplicate subjects in cohort")

    return df_fMRI_subset

def to_1_decimal_str(
    f:float
) -> str:
    return str(round(f, 1))

def compute_summary_features(
    df:pd.DataFrame, 
    utils:livingpark_utils.LivingParkUtils, 
    timepoint:str='baseline', 
    df_baseline : None | pd.DataFrame = None,
    index : dict = {"RAWHITE":"% Caucasian",
        "RABLACK":"% African-American",
        "RAASIAN":"% Asian",
        'HISPLAT':"% Hispanic",
        'SEX': "% Male",
        'HANDED': '% right-handed',
        'AGE_AT_VISIT': "Mean age, years",
        'EDUCYRS': 'Mean years of education',
        'PDXDUR':"Mean disease duration at baseline, days",
        'UPDRS_TOT_BASELINE' : "Mean MDS-UPDRS at baseline",
        'UPDRS_TOT_TIMEPOINT': 'Mean MDS-UPDRS at timepoint',
        'MCATOT': "Mean MoCA at baseline",
        'GDS_TOTAL': "Mean GDS at Baseline",
        'NHY': "Mean Hoehn-Yahr stage"}
) -> pd.DataFrame:
    """Compute cohort summary features to compare with papers Table 1. 

    Parameters
    ----------
    df : pd.DataFrame
        input dataframe
    utils : livingpark_utils.LivingParkUtils
        the notebook's LivingParkUtils instance
    timepoint : str
        timepoint of the cohort to compute summary measures (one of 'baseline', '1y', '2y' and '4y')
    df_baseline : None | pd.DataFrame
        required if timepoint other than baseline 
    index : dict
        columns names to input to new dataframe

    Returns
    -------
    pd.DataFrame
        dataframe with summary metrics
    """
    
    # Load necessary study files
    df_pd_history = load_ppmi_csv(utils, FILENAME_PD_HISTORY)
    df_demographics = load_ppmi_csv(utils, FILENAME_DEMOGRAPHICS)
    df_socio = load_ppmi_csv(utils, FILENAME_SOCIO)
    df_age = load_ppmi_csv(utils, FILENAME_AGE)
    df_disease_duration = disease_duration(utils.study_files_dir)
    
    # Necessary columns for demographic file
    cols_demo = ['PATNO', 'SEX', 'RAWHITE', 'HISPLAT', 'RABLACK', 'RAASIAN', 'HANDED']
    
    df_summary = df.merge(
        df_demographics[cols_demo],
        on=[COL_PAT_ID],
    )
    
    # Necessary columns for PD file
    cols_PD = ['PATNO','PDDXDT']
    
    df_summary = df_summary.merge(
        df_pd_history[cols_PD],
        on=[COL_PAT_ID],
    )
    
    # Necessary columns for social file 
    cols_socio = ['PATNO','EDUCYRS']
    
    df_summary = df_summary.merge(
        df_socio[cols_socio].drop_duplicates(subset=[COL_PAT_ID]),
        on=[COL_PAT_ID],
    )

    # Necessary columns for age file
    df_summary = df_summary.merge(df_age, on=[COL_PAT_ID, COL_VISIT_TYPE])
    df_summary = df_summary.drop_duplicates(subset=[COL_PAT_ID])
    
    # Conversion to binaries
    df_summary['HANDED'] = [1 if h==1 else 0 for h in df_summary['HANDED'].tolist()]

    # Baseline vs prediction timepoints
    if timepoint == 'baseline':
        df_summary['UPDRS_TOT_TIMEPOINT'] = df_summary['UPDRS_TOT']
        df_summary['UPDRS_TOT_BASELINE'] = df_summary['UPDRS_TOT']

        # Conversion of dates & binaries
        df_summary = df_summary.merge(df_disease_duration.drop_duplicates(subset=[COL_PAT_ID, COL_VISIT_TYPE]), 
                                      on = ['PATNO', 'EVENT_ID'])
        df_summary['PDXDUR'] = df_summary['PDXDUR'] / 12 * 366 

    else:
        df_summary['UPDRS_TOT_BASELINE'] = df_baseline[
        "UPDRS_TOT"][df_baseline[COL_PAT_ID].isin(
        df_summary[COL_PAT_ID].tolist())].tolist()
        df_summary['UPDRS_TOT_TIMEPOINT'] = df_summary['UPDRS_TOT']

        # Conversion of dates & binaries
        df_baseline = df_baseline.merge(df_disease_duration.drop_duplicates(subset=[COL_PAT_ID, COL_VISIT_TYPE]), 
                                      on = ['PATNO', 'EVENT_ID'])
        df_baseline['PDXDUR'] = df_baseline['PDXDUR'] / 12 * 366 

        df_summary['PDXDUR'] = df_baseline["PDXDUR"][df_baseline[COL_PAT_ID].isin(
        df_summary[COL_PAT_ID].tolist())].tolist()        
    
    # Drop unused columns    
    df_summary = df_summary[list(index.keys())]
    
    for keys in list(index.keys())[:6]:
        df_summary[keys] = df_summary[keys] * 100 
    
    df_summary_means = df_summary.mean().tolist()
    df_summary_stds = df_summary.std().tolist()
    
    df_summary_means = [to_1_decimal_str(mean) for mean in df_summary_means] 
    df_summary_stds = [" ± " + to_1_decimal_str(std) if i > 5 else '' for i, std in enumerate(df_summary_stds)] 
    df_summary_values = pd.DataFrame([df_summary_means, df_summary_stds], columns = df_summary.columns)
    df_summary_values = (df_summary_values.iloc[0] + df_summary_values.iloc[1]).T
    
    df_summary_values = df_summary_values.rename(index=index)
    
    if timepoint=='baseline':
        df_summary_values.loc['Mean MDS-UPDRS at timepoint'] = '-'

    return df_summary_values

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
    df_cohort_2y::pd.DataFrame, 
    df_cohort_4y::pd.DataFrame
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