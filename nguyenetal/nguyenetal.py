"""Helper functions for Nguyen et al. notebooks."""
from pathlib import Path

import pandas as pd

import livingpark_utils
from .constants import (
    FILENAME_PARTICIPANT_STATUS,
    FILENAME_DEMOGRAPHICS,
    FILENAME_PD_HISTORY,
    FILENAME_AGE,
    FILENAME_MOCA,
    FILENAME_UPDRS1A,
    FILENAME_UPDRS1B,
    FILENAME_UPDRS2,
    FILENAME_UPDRS3,
    FILENAME_UPDRS4,
    FILENAME_GDS,
    FILENAME_FMRI_INFO,
    FILENAME_FMRI_INFO_ZIP,
    FILENAME_FMRI_METADATA
)

from .constants import (
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

from .constants import (
    COL_DATE_INFO,
    COL_DATE_BIRTH,
    COL_DATE_PD,
    FIELD_STRENGTH,
    MANUFACTURER,
    IMAGE_DESC,
    STATUS_PD,
    TR_VAL,
)

from .constants import (
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
        df_ppmi[COL_STATUS] = df_ppmi[COL_STATUS].map(IDA_STATUS_MAP)

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
        df_ppmi = mean_impute(df_ppmi, cols_to_impute)

    return df_ppmi


def load_ppmi_fmri(
    utils: livingpark_utils.LivingParkUtils,
    filename: str,
    from_ida_search: bool = False,
    convert_dates: bool = True,
    alternative_dir: str = ".",
    cols_to_impute: str | list | None = None,
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
        
    df_fmri = pd.read_excel(filepath)

    # convert IDA search results to the same format as other PPMI study files
    if from_ida_search:
        df_fmri = df_fmri.rename(columns=IDA_COLNAME_MAP)
        
        # convert visit code
        missing_keys = set(df_fmri[COL_STATUS]) - set(IDA_STATUS_MAP.keys())
        if len(missing_keys) != 0:
            raise RuntimeError(f"Missing keys in conversion map: {missing_keys}")
        df_fmri[COL_STATUS] = df_fmri[COL_STATUS].map(IDA_STATUS_MAP)

    # convert subject IDs to integers
    # IDs should be all integers to be consistent
    # if they are strings the cohort_ID hash can change even if the strings are the same
    df_fmri[COL_PAT_ID] = pd.to_numeric(df_fmri[COL_PAT_ID], errors="coerce")
    invalid_subjects = df_fmri.loc[df_fmri[COL_PAT_ID].isna(), COL_PAT_ID].to_list()
    if len(invalid_subjects) > 0:
        print(
            f"Dropping {len(invalid_subjects)} subjects with non-integer IDs"
            # f": {invalid_subjects}"
        )
        df_fmri = df_fmri.loc[~df_fmri[COL_PAT_ID].isin(invalid_subjects)]

    if convert_dates:
        df_fmri = convert_date_cols(df_fmri, **kwargs)

    if cols_to_impute is not None:
        df_fmri = mean_impute(df_fmri, cols_to_impute)

    return df_fmri

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
    df_fMRI_subset = df_fMRI_subset.merge(df_status, on=[COL_PAT_ID, COL_STATUS])

    df_fMRI_subset = df_fMRI_subset.loc[
        (df_fMRI_subset[COL_STATUS].isin([status_groups])) & \
        (df_fMRI_subset[COL_IMAGING_PROTOCOL].str.contains(MANUFACTURER)) & \
        (df_fMRI_subset[COL_IMAGING_PROTOCOL].str.contains(FIELD_STRENGTH)) & \
        (df_fMRI_subset[COL_IMAGING_PROTOCOL].str.contains(TR_VAL)) & \
        (df_fMRI_subset[COL_IMAGE_DESC].isin(IMAGE_DESC))]

    #df_fMRI_subset = df_fMRI_subset.loc[
    #    (df_fMRI_subset[COL_FMRI]==1)]

    # warn if there are duplicates
    if df_fMRI_subset[COL_PAT_ID].nunique() != len(df_fMRI_subset[COL_PAT_ID]):
        print(f"WARNING: Duplicate subjects in cohort")

    return df_fMRI_subset