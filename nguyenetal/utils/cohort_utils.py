"""Helper functions for Nguyen et al. notebooks."""
from pathlib import Path
from functools import reduce

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
        (df_fMRI_subset[COL_IMAGING_PROTOCOL].str.contains(TR_VAL)) & \
        (df_fMRI_subset[COL_IMAGING_PROTOCOL].str.contains('TE=25.0')) & \
        (df_fMRI_subset[COL_IMAGE_DESC].isin(IMAGE_DESC))]

    #df_fMRI_subset = df_fMRI_subset.loc[
    #    (df_fMRI_subset[COL_FMRI]==1)]

    # warn if there are duplicates
    if df_fMRI_subset[COL_PAT_ID].nunique() != len(df_fMRI_subset[COL_PAT_ID]):
        print(f"WARNING: Duplicate subjects in cohort")

    return df_fMRI_subset


def get_scores_dataframe(
    utils:livingpark_utils.LivingParkUtils,
    df_fMRI_subset: pd.DataFrame,
    use_clean: bool=True
    ) -> pd.DataFrame:
    """
    Compute assesment file with clinical scores to filter the cohort. 

    Parameters
    ----------
    utils : livingpark_utils.LivingParkUtils
        the notebook's LivingParkUtils instance

    df_fMRI_subset : pd.DataFrame
        dataframe of participants having an fMRI image
    
    use_clean : bool
        True to use cleaned UPDRS-III file

    Returns
    -------
    df_assessments : pd.DataFrame
        dataframe with all scores
    """

    cols_for_merge = [COL_PAT_ID, COL_DATE_INFO, COL_VISIT_TYPE]

    # Load necessary files
    df_updrs1a = load_ppmi_csv(utils, FILENAME_UPDRS1A, convert_int = [COL_UPDRS1A])
    df_updrs1b = load_ppmi_csv(utils, FILENAME_UPDRS1B, convert_int = [COL_UPDRS1B])
    df_updrs2 = load_ppmi_csv(utils, FILENAME_UPDRS2, convert_int = [COL_UPDRS2])

    if use_clean:
        df_updrs3 = load_ppmi_csv(utils, FILENAME_UPDRS3_CLEAN, convert_int = [COL_UPDRS3])
    else: 
        df_updrs3 = load_ppmi_csv(utils, FILENAME_UPDRS3, convert_int = [COL_UPDRS3])
        
    df_updrs4 = load_ppmi_csv(utils, FILENAME_UPDRS4, convert_int = [COL_UPDRS4], cols_to_impute=COL_UPDRS4)

    df_moca = load_ppmi_csv(utils, FILENAME_MOCA, convert_int = [COL_MOCA])
    df_gds = load_ppmi_csv(utils, FILENAME_GDS)

    # Sum UPDRS IA and IB scores
    df_updrs1 = df_updrs1a.merge(df_updrs1b, on=cols_for_merge)
    df_updrs1[COL_UPDRS1] = df_updrs1.loc[:, [COL_UPDRS1A, COL_UPDRS1B]].sum(axis="columns")

    # Drop unused UPDRSIII scores (only ON medication)
    df_updrs3 = df_updrs3.drop(df_updrs3.index[(df_updrs3['PAG_NAME'] == 'NUPDRS3') & \
                             (df_updrs3['EVENT_ID'].isin(['V04', 'V06', 'V08', 'V10', 'V12', 'V13', 'V15']))])

    # Select UPDRS columns to merge
    df_updrs1 = df_updrs1.loc[:, cols_for_merge + [COL_UPDRS1]]
    df_updrs2 = df_updrs2.loc[:, cols_for_merge + [COL_UPDRS2]]
    df_updrs3 = df_updrs3.loc[:, cols_for_merge + [COL_UPDRS3, 'NHY', 'PAG_NAME', 'PDSTATE']]
    df_updrs4 = df_updrs4.loc[:, cols_for_merge + [COL_UPDRS4]]

    # Compute GDS total score 
    gds_cols = df_gds.columns[['GDS' in strcol for strcol in df_gds.columns]].tolist()
    df_gds['GDS_TOTAL'] = df_gds[gds_cols].sum(axis=1)
    df_gds = df_gds.loc[:, cols_for_merge + ['GDS_TOTAL']]

    # Select MOCA columns to merge
    df_moca = df_moca.loc[:, cols_for_merge + [COL_MOCA]]

    # Merge 
    df_assessments_all = reduce(
        lambda df1, df2: df1.merge(df2, on=cols_for_merge, how="outer"),
        [df_updrs2, df_updrs3, df_updrs1, df_updrs4, df_moca, df_gds],
    ).drop_duplicates()

    # Compute TOTAL UPDRS SCORE 
    updrs_cols = [COL_UPDRS1, COL_UPDRS2, COL_UPDRS3, COL_UPDRS4]
    df_assessments_all['UPDRS_TOT'] = df_assessments_all[updrs_cols].sum(axis=1)
       
    # Only keep cohort participants
    df_cohort_assessments = df_assessments_all.loc[
        df_assessments_all[COL_PAT_ID].isin(df_fMRI_subset[COL_PAT_ID])]

    # Drop participants that don't have UPDRS III Score
    df_cohort_assessments = df_cohort_assessments.dropna(subset=['NP3TOT'])

    return df_cohort_assessments


def to_1_decimal_str(
    f:float
) -> str:
    return str(round(f, 1))

def get_cohort_baseline(
    df_fMRI_subset:pd.DataFrame, 
    df_cohort_assessments:pd.DataFrame
):
    '''
    Function to filter fMRI and score files to keep only subjects who had a an fMRI and a UPDRS-III score at the same visit. 

    Parameters
    ----------
        df_fMRI_subset: pd.DataFrame
            DataFrame obtained with get_fMRI_cohort.

        df_cohort_assessments: pd.DataFrame
            DataFrame with scores of the participants with an fMRI image. 

    Returns
    -------
        df_global_cohort_baseline: pd.DataFrame
            DataFrame with only selected participants and sessions.
    '''
    df_fMRI_cohort = pd.DataFrame()
    # Go through score dataframe and for each line, check if participant name and visit name are in fMRI file
    for i in range(len(df_cohort_assessments)):
        df_fMRI_cohort = pd.concat([df_fMRI_cohort, 
                            df_fMRI_subset[df_fMRI_subset[COL_PAT_ID] == df_cohort_assessments.iloc[i][COL_PAT_ID]]\
                            [df_fMRI_subset[COL_VISIT_TYPE] == df_cohort_assessments.iloc[i][COL_VISIT_TYPE]]]
                            )
        
    df_scores_cohort = pd.DataFrame()
    # Same for fMRI dataset
    for i in range(len(df_fMRI_subset)):
        df_scores_cohort = pd.concat([df_scores_cohort, 
                        df_cohort_assessments[df_cohort_assessments[COL_PAT_ID] == df_fMRI_subset.iloc[i][COL_PAT_ID]]\
                        [df_cohort_assessments[COL_VISIT_TYPE] == df_fMRI_subset.iloc[i][COL_VISIT_TYPE]]]
                        )

    fMRI_cols_to_include = ['PATNO', 'Sex','COHORT_DEFINITION','EVENT_ID', 'INFODT', 'Age', 
                        'Description', 'Imaging Protocol', 'Image ID']
    scores_cols_to_include = ['PATNO', 'EVENT_ID','INFODT_SCORE', 'PDSTATE', 'PAG_NAME' ,'NP2PTOT', 'NP3TOT', 'NP1RTOT+NP1PTOT',
           'NP4TOT', 'NHY','MCATOT', 'GDS_TOTAL', 'UPDRS_TOT']
    
    df_fMRI_cohort = df_fMRI_cohort.loc[:, fMRI_cols_to_include]
    df_scores_cohort['INFODT_SCORE'] = df_scores_cohort['INFODT']
    df_scores_cohort = df_scores_cohort.loc[:, scores_cols_to_include]
    
    # Merge important columns from both datasets
    df_global_cohort = df_fMRI_cohort.merge(df_scores_cohort, on=[COL_PAT_ID, COL_VISIT_TYPE])
    df_global_cohort = df_global_cohort.sort_values(by=['PATNO','INFODT'])

    df_global_cohort_baseline = df_global_cohort.drop_duplicates(subset=COL_PAT_ID)
    df_global_cohort_baseline = df_global_cohort_baseline[
        df_global_cohort_baseline[COL_DATE_INFO] < pd.Timestamp(2020, 1, 1, 12)
        ] # Removed due to the date of the study

    return df_global_cohort_baseline

def get_cohort_prediction(
    df_global_cohort_baseline:pd.DataFrame, 
    df_cohort_assessments:pd.DataFrame
):
    '''
    Function to filter score files to keep only subjects who where included in baseline cohort and check if these participants 
    also had a score 1 year, 2 years and 4 years after the baseline visit. 

    Parameters
    ----------
        df_global_cohort_baseline: pd.DataFrame
            DataFrame obtained with get_cohort_baseline.

        df_cohort_assessments: pd.DataFrame
            DataFrame with scores of the participants with an fMRI image. 

    Returns
    -------
        df_global_1y: pd.DataFrame
            DataFrame with only selected participants and sessions for 1y.
        df_global_2y: pd.DataFrame
            DataFrame with only selected participants and sessions for 2y.
        df_global_4y: pd.DataFrame
            DataFrame with only selected participants and sessions for 4y.

    '''
    # DF with outcome scores for every participants selected at baseline
    df_global_cohort_pred = df_cohort_assessments[df_cohort_assessments\
                                                  [COL_PAT_ID].isin(df_global_cohort_baseline[COL_PAT_ID].tolist())]
    
    # Filter by date due to the date of publication of the paper. 
    df_global_cohort_pred = df_global_cohort_pred[df_global_cohort_pred[COL_DATE_INFO] < pd.Timestamp(2020, 1, 1, 12)]
    
    # Event taken as Baseline 
    df_global_cohort_pred['BASELINE_DATE'] = [df_global_cohort_baseline['INFODT_SCORE']\
                    [df_global_cohort_baseline[COL_PAT_ID] == df_global_cohort_pred[COL_PAT_ID].iloc[i]].iloc[0] \
                                 for i in range(len(df_global_cohort_pred))]
    
    df_global_cohort_pred['BASELINE_DATE'] = df_global_cohort_pred['BASELINE_DATE'].dt.strftime('%y-%m')
    df_global_cohort_pred['BASELINE_DATE'] = pd.to_datetime(df_global_cohort_pred['BASELINE_DATE'], format='%y-%m')
    df_global_cohort_pred['INFODT'] = df_global_cohort_pred['INFODT'].dt.strftime('%y-%m')
    df_global_cohort_pred['INFODT'] = pd.to_datetime(df_global_cohort_pred['INFODT'], format='%y-%m')

    df_global_1y = pd.DataFrame()
    df_global_2y = pd.DataFrame()
    df_global_4y = pd.DataFrame()
    
    for sub in df_global_cohort_baseline['PATNO'].tolist():
        df_subject = df_global_cohort_pred.loc[(df_global_cohort_pred['PATNO'] == sub)]
    
        if len(df_subject)==0:
            continue

        # Check for score at different time after baseline
        df_global_1y = pd.concat([df_global_1y, 
                df_subject.loc[((df_subject['INFODT'] >= (df_subject.iloc[0]['BASELINE_DATE'] + dt.timedelta(days=365 - 62))) &\
                            (df_subject['INFODT'] <= (df_subject.iloc[0]['BASELINE_DATE'] + dt.timedelta(days=365 + 62))))]])
    
        df_global_2y = pd.concat([df_global_2y, 
                df_subject.loc[((df_subject['INFODT'] >= (df_subject.iloc[0]['BASELINE_DATE'] + dt.timedelta(days=2*365 - 62))) &\
                            (df_subject['INFODT'] <= (df_subject.iloc[0]['BASELINE_DATE'] + dt.timedelta(days=2*365 + 62))))]])
        
        df_global_4y = pd.concat([df_global_4y, 
                df_subject.loc[((df_subject['INFODT'] >= (df_subject.iloc[0]['BASELINE_DATE'] + dt.timedelta(days=4*365 - 62))) &\
                            (df_subject['INFODT'] <= (df_subject.iloc[0]['BASELINE_DATE'] + dt.timedelta(days=4*365 + 62))))]])
            
    df_global_1y = df_global_1y.drop_duplicates(['PATNO'])
    df_global_1y = df_global_1y.merge(df_global_cohort_baseline[['PATNO','Sex']], on=['PATNO'])
    
    df_global_2y = df_global_2y.drop_duplicates(['PATNO'])
    df_global_2y = df_global_2y.merge(df_global_cohort_baseline[['PATNO','Sex']], on=['PATNO'])
    
    df_global_4y = df_global_4y.drop_duplicates(['PATNO'])
    df_global_4y = df_global_4y.merge(df_global_cohort_baseline[['PATNO','Sex']], on=['PATNO'])

    return df_global_1y, df_global_2y, df_global_4y

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

    return df_summary_values, df_summary

def plot_summary_cohort(
    df_cohort_baseline,
    df_cohort_1y, 
    df_cohort_2y,
    df_cohort_4y,
    utils
):
    '''
    Compute summary features of the cohort at different timepoints based on the figure of the original paper. 

    Parameters
    ----------
        df_cohort_baseline: pd.DataFrame

        df_cohort_1y: pd.DataFrame

        df_cohort_2y: pd.DataFrame

        df_cohort_4y: pd.DataFrame

        utils : livingpark_utils.LivingParkUtils
        the notebook's LivingParkUtils instance

    Returns
    -------
        df_allyears_summary: pd.DataFrame
    '''
    df_allyears_summary = pd.DataFrame(columns = [('Baseline', 'Original'), ('Baseline', 'Replication'),
                                                  ('Year 1', 'Original'), ('Year 1', 'Replication'), 
                                                  ('Year 2', 'Original'), ('Year 2', 'Replication'), 
                                                  ('Year 4', 'Original'), ('Year 4', 'Replication')])
    
    df_allyears_summary[('Baseline', 'Original')]=['95.1','2.4','3.7','1.2','67.0','89.0','62.1 ± 9.8',
     '15.6 ± 3.0','770 ± 565','33.9 ± 15.8','-','26.7 ± 2.8','5.4 ± 1.4','1.8 ± 0.5']
    df_allyears_summary[('Year 1', 'Original')] = ['94.4','1.9','5.6','0','68.5','85.2','61.9 ± 10.3',
     '15.1 ± 3.2','808 ± 576','38.0 ± 20.9','39.2 ± 21.6','26.9 ± 3.2','5.4 ± 1.6','1.8 ± 0.5']
    df_allyears_summary[('Year 2', 'Original')] = ['97.8','0','4.4','0','82.2','88.9','63.6 ± 9.2',
     '15.1 ± 3.3','771 ± 506','40.2 ± 18.2','40.9 ± 18.5','26.7 ± 3.5','5.4 ± 1.2','1.8 ± 0.5']
    df_allyears_summary[('Year 4', 'Original')] = ['97.0','0','3.0','0','75.8','87.9','59.5 ± 11.0',
        '15.0 ± 3.4','532 ± 346','34.9 ± 15.7','35.9 ± 16.5','27.5 ± 2.3','5.4 ± 1.7','1.7 ± 0.5']
    
    df_allyears_summary[('Baseline', 'Replication')] = compute_summary_features(df_cohort_baseline, utils,
                                                                    'baseline')[0].tolist()
    df_allyears_summary[('Year 1', 'Replication')] = compute_summary_features(df_cohort_1y, utils,
                                                                    '1Y', df_cohort_baseline)[0].tolist()
    df_allyears_summary[('Year 2', 'Replication')] = compute_summary_features(df_cohort_2y, utils,
                                                                    '2Y', df_cohort_baseline)[0].tolist()
    df_allyears_summary[('Year 4', 'Replication')] = compute_summary_features(df_cohort_4y, utils,
                                                                    '4Y', df_cohort_baseline)[0].tolist()
    
    df_allyears_summary.index = compute_summary_features(df_cohort_baseline, utils,
                                                                    'baseline')[0].index
    
    df_allyears_summary.loc['Number of subject'] = [82, len(df_cohort_baseline), 53, len(df_cohort_1y), 
                                          45, len(df_cohort_2y), 33, len(df_cohort_4y)]
    
    df_allyears_summary.columns = pd.MultiIndex.from_tuples(df_allyears_summary.columns)
    
    return df_allyears_summary

