"""Constant variables for Nguyen et al. notebooks."""

# PPMI file names
FILENAME_DEMOGRAPHICS = "Demographics.csv"
FILENAME_PD_HISTORY = "PD_Diagnosis_History.csv"
FILENAME_AGE = "Age_at_visit.csv"
FILENAME_SOCIO = "Socio-Economics.csv"
FILENAME_PARTICIPANT_STATUS = "Participant_Status.csv"
FILENAME_MOCA = "Montreal_Cognitive_Assessment__MoCA_.csv"
FILENAME_UPDRS1A = "MDS-UPDRS_Part_I.csv"
FILENAME_UPDRS1B = "MDS-UPDRS_Part_I_Patient_Questionnaire.csv"  # patient questionnaire
FILENAME_UPDRS2 = "MDS_UPDRS_Part_II__Patient_Questionnaire.csv"
FILENAME_UPDRS3 = "MDS-UPDRS_Part_III.csv"
FILENAME_UPDRS4 = "MDS-UPDRS_Part_IV__Motor_Complications.csv"
FILENAME_GDS = "Geriatric_Depression_Scale__Short_Version_.csv"

# other file names
FILENAME_FMRI_INFO = "PPMI_rs-fMRI_Data-and-Methods/PPMI_fs-fMRI_dataReport_spreadsheet_FINAL_20191121.xlsx"
FILENAME_FMRI_INFO_ZIP = "PPMI_rs-fMRI_Data-and-Methods.zip"
FILENAME_FMRI_METADATA = "fMRI_info.csv"


# useful column names
COL_PAT_ID = "PATNO"
COL_STATUS = "COHORT_DEFINITION"
COL_VISIT_TYPE = "EVENT_ID"
COL_ENROLLMENT = 'ENROLL_STATUS'
COL_DATE_INFO = "INFODT"
COL_DATE_BIRTH = "BIRTHDT"
COL_DATE_PD = "PDDXDT"
COLS_DATE = [
    COL_DATE_INFO,
    "LAST_UPDATE",
    "ORIG_ENTRY",
    COL_DATE_BIRTH,
    COL_DATE_PD,
]
COL_PD_STATE = "PDSTATE"
COL_VISIT_TYPE = "EVENT_ID"
COL_PD_STATE = "PDSTATE"
COL_AGE = "AGE_AT_VISIT"
COL_SEX = "SEX"
COL_EDUCATION = "EDUCYRS"
COL_MOCA = "MCATOT"
COL_UPDRS1A = "NP1RTOT"
COL_UPDRS1B = "NP1PTOT"
COL_UPDRS1 = f"{COL_UPDRS1A}+{COL_UPDRS1B}"
COL_UPDRS2 = "NP2PTOT"
COL_UPDRS3 = "NP3TOT"
COL_UPDRS4 = "NP4TOT"

COL_FOLLOWUP = "is_followup"
COL_IMAGING_PROTOCOL = "Imaging Protocol" 
COL_IMAGE_DESC = "Description"

# codes for COHORT_DEFINITION field
STATUS_PD = "Parkinson's Disease"
STATUS_HC = "Healthy Control"

# codes for EVENT_ID field
REGEX_VISIT_FOLLOWUP = "^V((0[4-9])|(1[0-9])|(20))$"  # V04-V20

# codes for SEX field
SEX_FEMALE = 0
SEX_MALE = 1

# Acquisition parameters
FIELD_STRENGTH = "Field Strength=3.0"
MANUFACTURER = "SIEMENS"
IMAGE_DESC = [
    "ep2d_RESTING_STATE", 
    "ep2d_bold_rest",
]
TR_VAL = "TR=2400.0"
STATUS_MED = "ON"

# codes for SEX field
SEX_FEMALE = 0
SEX_MALE = 1

# column names/values obtained from searching the Image and Data Archive (IDA)
# are different from those used in the PPMI csv study files
# so they need to be remapped
IDA_COLNAME_MAP = {
    "Subject ID": COL_PAT_ID,
    "Visit": COL_VISIT_TYPE,
    "Study Date": COL_DATE_INFO,
    "Research Group": COL_STATUS
}
IDA_VISIT_MAP = {
    "Baseline": "BL",
    "Screening": "SC",
    "Month 6": "V02",
    "Month 12": "V04",
    "Month 24": "V06",
    "Month 36": "V08",
    "Month 48": "V10",
    "Symptomatic Therapy": "ST",
    "Unscheduled Visit 01": "U01",
    "Unscheduled Visit 02": "U02",
    "Premature Withdrawal": "PW",
}
IDA_STATUS_MAP = {
    "PD": "Parkinson's Disease",
    "Control":"Healthy Control",
    "SWEDD": "SWEDD",
    "Prodromal": "Prodromal",
    "GenCohortPD":"Parkinson's Disease",
    "GenCohortUnaff":"Prodromal"
}
